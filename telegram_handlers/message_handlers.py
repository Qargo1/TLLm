# telegram_handlers/message_handlers.py
import asyncio
import html
import re # Для парсинга ответа рефлексии
from typing import List, Dict # Добавляем типизацию

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction, ParseMode

from config import (
    logger, LTM_RETRIEVAL_COUNT, SCENARIOS, # Добавляем SCENARIOS
    REFLECTION_INTERVAL_MESSAGES, REFLECTION_PROMPT_TEMPLATE,
    # Параметры генерации для рефлексии
    REFLECTION_MAX_NEW_TOKENS, REFLECTION_TEMPERATURE,
    REFLECTION_TOP_P, REFLECTION_TOP_K, REFLECTION_REPETITION_PENALTY
)
# Импортируем нужную функцию генерации (нужно будет создать/адаптировать)
from llm.llama_loader import generate_with_params, tokenizer, llm_model
from memory.long_term import LongTermMemory
from memory.vector_store import VectorStoreManager
from llm.prompt_builder import build_prompt_for_llama3
from persistence.chat_manager import ChatManager
from persistence.chat_state import ChatState
# Импортируем для проверки типов
from memory.short_term import ShortTermMemory, Message
from persistence.database import save_chat_state

# --- Фоновая задача Рефлексии ---

async def _run_reflection(
    chat_state: ChatState,
    ltm: LongTermMemory,
    tokenizer, # Передаем токенизатор
    context: ContextTypes.DEFAULT_TYPE):
    """Асинхронная задача для анализа диалога и обновления LTM/промпта."""
    chat_id = chat_state.chat_id
    logger.info(f"Запуск рефлексии для чата {chat_id}...")

    if not context.application.bot_data.get('llm_loaded', False) or llm_model is None:
        logger.warning(f"Рефлексия для чата {chat_id} отменена: LLM не загружена.")
        return
    if not context.application.bot_data.get('embedding_model_loaded', False):
        logger.warning(f"Рефлексия для чата {chat_id} отменена: модель эмбеддингов не загружена (факты не будут добавлены в LTM).")
        # Можно продолжить только с обновлением промпта, но лучше прервать

    # 1. Получаем недавнюю историю (например, всю STM)
    # Ограничим размер истории для рефлексии, чтобы не перегружать LLM
    history_for_reflection: List[Message] = chat_state.get_stm_messages()
    if not history_for_reflection:
        logger.warning(f"Рефлексия для чата {chat_id}: нет истории в STM для анализа.")
        return

    # Форматируем историю для промпта рефлексии
    recent_history_text = "\n".join([f"[{msg['role'].capitalize()}] {msg['content']}" for msg in history_for_reflection])

    # 2. Формируем промпт для LLM рефлексии
    current_dynamic_prompt = chat_state.system_prompt or "" # Берем текущий дин. промпт
    reflection_prompt_filled = REFLECTION_PROMPT_TEMPLATE.format(
        dynamic_system_prompt=current_dynamic_prompt,
        recent_history=recent_history_text
    )
    # Преобразуем строку промпта в формат messages для generate_with_params
    # Так как наш шаблон уже имитирует диалог System/User/Assistant,
    # можно просто передать его как одно сообщение user'а под системной оберткой (или без нее).
    # Но проще использовать apply_chat_template, если он есть у модели.
    # Пока сделаем проще: передадим как есть (модели Llama3 должны справиться)
    # TODO: Уточнить, как лучше передавать такой мета-промпт в generate_with_params
    reflection_messages = [{"role": "user", "content": reflection_prompt_filled}] # Упрощенный вариант

    logger.debug(f"Промпт рефлексии для чата {chat_id}:\n{reflection_prompt_filled[:500]}...") # Логируем начало

    # 3. Вызываем LLM для рефлексии с другими параметрами
    try:
        reflection_result = await asyncio.to_thread(
            generate_with_params, # Используем новую функцию
            messages=reflection_messages,
            max_new_tokens=REFLECTION_MAX_NEW_TOKENS,
            temperature=REFLECTION_TEMPERATURE,
            top_p=REFLECTION_TOP_P,
            top_k=REFLECTION_TOP_K,
            repetition_penalty=REFLECTION_REPETITION_PENALTY
        )
        logger.debug(f"Результат рефлексии для чата {chat_id}:\n{reflection_result}")

    except Exception as e:
        logger.error(f"Ошибка LLM во время рефлексии для чата {chat_id}: {e}", exc_info=True)
        return # Прерываем рефлексию при ошибке

    # 4. Парсим результат
    extracted_facts: List[str] = []
    updated_state_text: Optional[str] = None

    try:
        # Ищем факты
        for line in reflection_result.splitlines():
            line = line.strip()
            if line.upper().startswith("FACT:"):
                fact = line[len("FACT:"):].strip()
                if fact and fact.lower() != "none":
                    extracted_facts.append(fact)

        # Ищем обновленное состояние
        match = re.search(r"<UPDATED_STATE>(.*?)</UPDATED_STATE>", reflection_result, re.IGNORECASE | re.DOTALL)
        if match:
            updated_state_text = match.group(1).strip()

    except Exception as e:
        logger.error(f"Ошибка парсинга результата рефлексии для чата {chat_id}: {e}\nРезультат:\n{reflection_result}", exc_info=True)
        # Не прерываем, попробуем сохранить то, что удалось распарсить

    # 5. Обновляем LTM (факты)
    if extracted_facts and context.application.bot_data.get('embedding_model_loaded', False):
        logger.info(f"Рефлексия для чата {chat_id}: извлечено {len(extracted_facts)} фактов для LTM.")
        added_count = 0
        # Добавляем факты по одному (можно оптимизировать батчингом, если ltm.add_memories есть)
        for fact in extracted_facts:
            try:
                # Используем to_thread для синхронного add_memory
                success = await asyncio.to_thread(ltm.add_memory, fact)
                if success:
                    added_count += 1
                    logger.debug(f"Факт добавлен в LTM чата {chat_id}: '{fact[:50]}...'")
                else:
                     logger.warning(f"Не удалось добавить факт в LTM чата {chat_id}: '{fact[:50]}...'")
            except Exception as e:
                 logger.error(f"Ошибка добавления факта в LTM чата {chat_id}: {e}", exc_info=True)
        logger.info(f"Добавлено {added_count} из {len(extracted_facts)} фактов в LTM чата {chat_id}.")
    elif extracted_facts:
         logger.warning(f"Извлечено {len(extracted_facts)} фактов рефлексии для чата {chat_id}, но модель эмбеддингов не загружена.")


    # 6. Обновляем динамический промпт в ChatState
    state_changed = False
    if updated_state_text:
        # Проверяем, отличается ли он от текущего, чтобы не сохранять зря
        if updated_state_text != current_dynamic_prompt:
            logger.info(f"Рефлексия для чата {chat_id}: Обновление динамического промпта.")
            logger.debug(f"Новый дин. промпт для чата {chat_id}:\n{updated_state_text}")
            chat_state.update_dynamic_prompt(updated_state_text)
            state_changed = True
        else:
            logger.info(f"Рефлексия для чата {chat_id}: Динамический промпт не изменился.")
    else:
        logger.warning(f"Рефлексия для чата {chat_id}: Не удалось извлечь обновленное состояние (<UPDATED_STATE>).")

    # 7. Сохраняем ChatState, если были изменения
    if state_changed:
        try:
            # TODO: Сделать save_chat_state асинхронной?
            save_chat_state(chat_state)
            logger.info(f"Обновленное состояние чата {chat_id} после рефлексии сохранено.")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния чата {chat_id} после рефлексии: {e}", exc_info=True)

    logger.info(f"Рефлексия для чата {chat_id} завершена.")


# --- Основной Обработчик Сообщений ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения пользователя."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    message_text = update.message.text.strip()

    if not message_text:
        return

    # --- Проверки доступности сервисов ---
    if not context.application.bot_data.get('llm_loaded', False) or llm_model is None:
        logger.warning(f"LLM не загружена. Сообщение от {user_id} в чате {chat_id} не обработано.")
        # Не отвечаем пользователю, чтобы не прерывать возможное ожидание
        return
    if tokenizer is None:
        logger.error("Токенизатор LLM не загружен. Обработка сообщения невозможна.")
        await update.message.reply_text("Произошла внутренняя ошибка (компоненты LLM недоступны).")
        return

    chat_manager: ChatManager = context.application.bot_data.get('chat_manager')
    vector_store_manager: VectorStoreManager = context.application.bot_data.get('vector_store_manager')
    if not chat_manager or not vector_store_manager:
         logger.error(f"Менеджеры не найдены для чата {chat_id}")
         await update.message.reply_text("Произошла внутренняя ошибка (компоненты памяти недоступны).")
         return

    logger.info(f"Сообщение от {user_id} в чате {chat_id}: '{message_text[:50]}...'")

    # --- Получение/создание состояния чата ---
    try:
        # Используем кастомный контекст для удобства
        chat_state: ChatState = await context.chat_manager.get_chat_state(chat_id, user_id)
        # Обновляем время взаимодействия здесь
        chat_state.update_interaction_time()
    except Exception as e:
         logger.error(f"Ошибка при получении состояния чата {chat_id}: {e}", exc_info=True)
         await update.message.reply_text("Произошла ошибка при доступе к данным чата.")
         return

    # --- Проверка выбран ли сценарий ---
    if chat_state.scenario_id is None or chat_state.system_prompt is None:
        logger.warning(f"Для чата {chat_id} не выбран сценарий. Просим выбрать.")
        # Отправляем сообщение с кнопками выбора сценария (как в /start)
        keyboard = [
            [InlineKeyboardButton(f"{idx}. {info['name']}", callback_data=f"select_scenario_{idx}")] # Используем префикс из main.py
            for idx, info in SCENARIOS.items()
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Пожалуйста, сначала выбери сценарий:",
            reply_markup=reply_markup
        )
        return

    # --- Инициализация LTM ---
    ltm = None
    if chat_state.ltm_collection_id:
        try:
            ltm = LongTermMemory(
                chat_id=chat_id,
                user_id=user_id, # user_id не обязателен здесь, но для полноты
                vector_store=vector_store_manager
            )
            # Проверяем/исправляем имя коллекции
            if ltm.get_collection_name() != chat_state.ltm_collection_id:
                 logger.warning(f"Исправление ltm_collection_id для чата {chat_id}.")
                 ltm.collection_name = chat_state.ltm_collection_id
        except Exception as e:
             logger.error(f"Ошибка инициализации LTM для чата {chat_id}: {e}", exc_info=True)
             ltm = None # Продолжаем без LTM
    else:
        logger.error(f"У ChatState для чата {chat_id} отсутствует ltm_collection_id!")
        # Это не должно происходить, но на всякий случай

    # --- Извлечение из LTM ---
    retrieved_ltm: List[str] = []
    if ltm and context.application.bot_data.get('embedding_model_loaded', False):
        try:
            retrieved_ltm = await asyncio.to_thread(
                ltm.retrieve_relevant_memories,
                query_text=message_text,
                n=LTM_RETRIEVAL_COUNT
            )
            if retrieved_ltm:
                logger.info(f"Извлечено {len(retrieved_ltm)} факт(ов) из LTM для чата {chat_id}.")
        except Exception as e:
            logger.error(f"Ошибка при извлечении из LTM для чата {chat_id}: {e}", exc_info=True)
    elif ltm:
         logger.warning(f"Пропуск поиска в LTM для чата {chat_id} - модель эмбеддингов не загружена.")

    # --- Сборка промпта ---
    system_prompt = chat_state.system_prompt # Используем ДИНАМИЧЕСКИЙ промпт
    stm_history = chat_state.get_stm_messages()
    user_prompt = message_text
    try:
        messages_for_llm = build_prompt_for_llama3(
            system_prompt=system_prompt,
            ltm_memories=retrieved_ltm, # Передаем извлеченные факты
            stm_history=stm_history,
            user_prompt=user_prompt,
            tokenizer=tokenizer
        )
    except Exception as e:
        # ... (обработка ошибки сборки промпта) ...
        return

    # Показываем индикатор "печатает..."
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    llm_response = "Ошибка: Ответ не был сгенерирован."
    try:
        # --- Вызов LLM для генерации ответа ---
        # Используем generate_with_params с основными параметрами генерации
        # (Предполагается, что эта функция теперь существует в llama_loader)
        llm_response = await asyncio.to_thread(
            generate_with_params, # Новая функция
            messages=messages_for_llm,
            # Передаем основные параметры генерации из config
            max_new_tokens=context.application.bot_data.get('GENERATION_MAX_NEW_TOKENS', 300), # Пример получения из bot_data
            temperature=context.application.bot_data.get('GENERATION_TEMPERATURE', 0.75),
            top_p=context.application.bot_data.get('GENERATION_TOP_P', 0.9),
            top_k=context.application.bot_data.get('GENERATION_TOP_K', 50),
            repetition_penalty=context.application.bot_data.get('GENERATION_REPETITION_PENALTY', 1.1)
            # TODO: Перенести параметры генерации в bot_data при старте для удобства
        )

        # --- Обновление STM и Запуск Рефлексии ---
        if llm_response and "Ошибка генерации" not in llm_response and "Ошибка:" not in llm_response:
            # Обновляем STM (это также увеличит message_counter)
            chat_state.add_stm_entry(user_message=user_prompt, bot_message=llm_response)

            # Проверяем, пора ли запускать рефлексию
            if chat_state.message_counter >= REFLECTION_INTERVAL_MESSAGES:
                logger.info(f"Счетчик сообщений чата {chat_id} достиг {chat_state.message_counter}. Запуск рефлексии.")
                # Сбрасываем счетчик ПЕРЕД запуском задачи
                chat_state.message_counter = 0
                # Запускаем рефлексию в фоне
                asyncio.create_task(
                    _run_reflection(chat_state, ltm, tokenizer, context),
                    name=f"Reflection_{chat_id}"
                )
            # Сохраняем состояние (обновленную STM и, возможно, сброшенный счетчик)
            save_chat_state(chat_state) # TODO: Сделать асинхронной?
            logger.debug(f"Состояние чата {chat_id} сохранено после обработки сообщения.")

        # --- Отправка Ответа ---
        if llm_response and "Ошибка генерации" not in llm_response and "Ошибка:" not in llm_response:
            await update.message.reply_text(llm_response)
            logger.info(f"Ответ LLM для чата {chat_id} отправлен.")
        else:
             await update.message.reply_text("Я задумался и не смог ничего ответить, или произошла ошибка. Попробуешь перефразировать?")

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке сообщения чата {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("Произошла серьезная внутренняя ошибка при обработке вашего сообщения.")