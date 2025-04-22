# llm/prompt_builder.py
from typing import List, Dict, Optional

# Импортируем Message из STM для type hinting
from memory.short_term import Message
# Импортируем логгер и лимит контекста из config
from config import logger, GENERATION_MAX_NEW_TOKENS # Используем max_new_tokens для расчета лимита

# --- Константы ---
# Оставляем небольшой запас токенов для генерации ответа и спец. токенов шаблона
# Llama3 может иметь контекст 8k или 128k, но для локального запуска лучше сильно ограничить
# Будем ориентироваться на максимальный размер *входного* промпта
# Пример: Модель имеет контекст 4096. Мы хотим генерировать до 300 токенов.
# Значит, входной промпт не должен превышать ~4096 - 300 - (запас ~100) = 3696
# Вместо жесткого значения, можно использовать размер контекста модели, если он доступен
# Пока зададим константой, но можно получать из config или tokenizer.model_max_length
# Уменьшим значение по умолчанию для большей надежности на слабых машинах
MAX_INPUT_TOKENS = 2048 # Максимум токенов для ВХОДНОГО промпта

# --- Вспомогательная функция оценки токенов ---
def estimate_token_count(text: str, tokenizer) -> int:
    """Оценивает количество токенов с использованием предоставленного токенизатора."""
    if not tokenizer:
        # Очень грубая оценка, если токенизатор недоступен
        # Лучше вернуть большое число, чтобы вызвать обрезку, чем недооценить
        logger.warning("Токенизатор недоступен для estimate_token_count, используется грубая оценка.")
        return len(text) // 2 # Более консервативная оценка
    try:
        # Используем encode, так как он просто возвращает список ID
        return len(tokenizer.encode(text, add_special_tokens=False)) # Не добавляем спец.токены здесь
    except Exception as e:
        logger.error(f"Ошибка при оценке токенов: {e}. Используется грубая оценка.")
        # Фоллбэк
        return len(text) // 2

# --- Основная функция сборки промпта ---
def build_prompt_for_llama3(
    system_prompt: Optional[str],
    ltm_memories: List[str],
    stm_history: List[Message],
    user_prompt: str,
    tokenizer # Передаем токенизатор для оценки длины
) -> List[Message]:
    """
    Собирает историю сообщений для Llama 3 Chat Template, включая LTM.
    Обрезает STM, если общая длина превышает MAX_INPUT_TOKENS.

    Args:
        system_prompt (Optional[str]): Динамический системный промпт.
        ltm_memories (List[str]): Список релевантных воспоминаний из LTM.
        stm_history (List[Message]): История сообщений из STM.
        user_prompt (str): Текущее сообщение пользователя.
        tokenizer: Загруженный токенизатор.

    Returns:
        List[Message]: Список сообщений для передачи в apply_chat_template.
                       Может быть пустым, если даже user_prompt не помещается.
    """
    messages: List[Message] = []
    current_tokens = 0

    # 1. Учитываем токены для ответа
    # Оставляем место для max_new_tokens + небольшой буфер (~50-100 токенов) для спец. символов шаблона
    # Важно: GENERATION_MAX_NEW_TOKENS берется из config и должен быть актуальным для текущего вызова
    token_limit_for_prompt = MAX_INPUT_TOKENS
    # logger.debug(f"Общий лимит токенов для промпта: {token_limit_for_prompt}")

    # 2. Добавляем System Prompt и LTM
    system_content = system_prompt or ""
    if ltm_memories:
        # Форматируем LTM с разделителем
        ltm_context = "\n\n--- Relevant Past Memories ---\n"
        for mem in ltm_memories:
            ltm_context += f"- {mem}\n"
        system_content += ltm_context.rstrip() # Добавляем к системному промпту

    if system_content:
        system_tokens = estimate_token_count(system_content, tokenizer)
        if system_tokens < token_limit_for_prompt:
             messages.append({"role": "system", "content": system_content})
             current_tokens += system_tokens
             logger.debug(f"Добавлен System+LTM: {system_tokens} токенов.")
        else:
             # Если даже System+LTM не влезают, это проблема
             logger.warning(f"System prompt + LTM ({system_tokens} токенов) превышает лимит {token_limit_for_prompt}. Промпт может быть неполным.")
             # Обрезаем сам system_content (грубо) - не лучший вариант, но предотвращает падение
             # TODO: Нужна более умная стратегия обрезки системного промпта/LTM
             allowed_system_text = tokenizer.decode(tokenizer.encode(system_content)[:token_limit_for_prompt - 50]) # Оставим запас
             messages.append({"role": "system", "content": allowed_system_text + "\n[... LTM/System Prompt Truncated ...]"})
             current_tokens = estimate_token_count(messages[0]['content'], tokenizer)


    # 3. Добавляем User Prompt (последнее сообщение пользователя)
    user_prompt_tokens = estimate_token_count(user_prompt, tokenizer)
    if current_tokens + user_prompt_tokens >= token_limit_for_prompt:
         # Если даже system + user не влезают, обрезать нечего, возвращаем только их (или ошибку)
         logger.warning(f"System prompt + User prompt ({current_tokens + user_prompt_tokens} токенов) уже превышают лимит {token_limit_for_prompt}. STM будет пустой.")
         # Оставляем только system и user
         messages.append({"role": "user", "content": user_prompt})
         return messages # Возвращаем только system и user

    # Рассчитываем доступные токены для STM
    available_tokens_for_stm = token_limit_for_prompt - current_tokens - user_prompt_tokens
    logger.debug(f"Доступно токенов для STM: {available_tokens_for_stm}")

    # 4. Добавляем STM (с обрезкой)
    truncated_stm: List[Message] = []
    current_stm_tokens = 0
    original_stm_count = len(stm_history)

    # Идем с конца (новые сообщения важнее)
    for msg in reversed(stm_history):
        msg_content = msg.get("content", "")
        msg_tokens = estimate_token_count(msg_content, tokenizer)

        if current_stm_tokens + msg_tokens <= available_tokens_for_stm:
            truncated_stm.insert(0, msg) # Добавляем в начало, чтобы сохранить порядок
            current_stm_tokens += msg_tokens
        else:
            # Место закончилось
            logger.info(f"Достигнут лимит токенов для STM. Сохранено {len(truncated_stm)} из {original_stm_count} сообщений ({current_stm_tokens} токенов).")
            break # Прерываем цикл

    # Добавляем обрезанную STM в итоговый список
    messages.extend(truncated_stm)

    # 5. Добавляем User Prompt в конец
    messages.append({"role": "user", "content": user_prompt})

    final_token_estimate = sum(estimate_token_count(m["content"], tokenizer) for m in messages)
    logger.debug(f"Финальный промпт: {len(messages)} сообщений, ~{final_token_estimate} токенов.")

    # Логируем финальную структуру сообщений (без токенизации)
    # logger.debug(f"Собранная структура сообщений для LLM: {messages}") # Уже логируется выше

    return messages