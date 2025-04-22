# llm/llama_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Убрали pipeline
from typing import List, Dict, Optional

from config import (
    MODEL_NAME_OR_PATH, MODEL_REVISION, DEVICE, TORCH_DTYPE,
    # Убираем импорт параметров генерации по умолчанию, они будут передаваться
    # GENERATION_MAX_NEW_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P, GENERATION_TOP_K,
    logger
)
import time
import gc

# --- Глобальные переменные ---
model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None

# --- Загрузка Модели ---
def load_llm_model() -> tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Загружает LLM модель и токенизатор. Возвращает (model, tokenizer) или (None, None).
    """
    global model, tokenizer
    if model is not None and tokenizer is not None:
        logger.info("Модель и токенизатор LLM уже загружены.")
        return model, tokenizer

    logger.info(f"Загрузка LLM модели: {MODEL_NAME_OR_PATH} (Ревизия: {MODEL_REVISION})")
    logger.info(f"Используемое устройство: {DEVICE} | Тип данных Torch: {TORCH_DTYPE}")

    load_start_time = time.time()
    quantization_config = None
    model_kwargs = {"revision": MODEL_REVISION}

    # --- Квантизация (опционально) ---
    # TODO: Реализовать при необходимости, добавить флаг USE_QUANTIZATION в config.py
    # try:
    #     if DEVICE == "cuda" and USE_QUANTIZATION:
    #         logger.info("Настройка 4-битной квантизации...")
    #         quantization_config = BitsAndBytesConfig(...)
    #         model_kwargs["quantization_config"] = quantization_config
    #         model_kwargs["device_map"] = "auto" # accelerate для квантизации
    # except NameError: # Если USE_QUANTIZATION не определен
    #     logger.debug("Параметр USE_QUANTIZATION не установлен, квантизация не используется.")
    # except ImportError:
    #     logger.warning("bitsandbytes не найден, квантизация недоступна.")
    # except Exception as e:
    #     logger.error(f"Ошибка настройки квантизации: {e}. Загрузка без квантизации.")
    # ---

    # --- Стандартная загрузка ---
    if "quantization_config" not in model_kwargs:
         model_kwargs["torch_dtype"] = TORCH_DTYPE
         if DEVICE == 'cuda':
             # accelerate может помочь с распределением большой модели на GPU
             model_kwargs["device_map"] = "auto"
             logger.info("Используется device_map='auto' для GPU.")
         # Для CPU device_map не нужен, модель загрузится на CPU по умолчанию

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, revision=MODEL_REVISION)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Установлен tokenizer.pad_token = eos_token ('{tokenizer.pad_token}')")
            else:
                 logger.error("У токенизатора нет ни pad_token, ни eos_token!")
                 # Это критично, генерация может сломаться
                 raise ValueError("Токенизатор должен иметь pad_token или eos_token")
        # Гарантируем наличие pad_token_id
        if tokenizer.pad_token_id is None:
             tokenizer.pad_token_id = tokenizer.eos_token_id # Llama обычно имеет eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            **model_kwargs,
            # trust_remote_code=True # Иногда требуется
        )

        # Если модель не распределилась через device_map (например, CPU или ошибка)
        if not hasattr(model, 'hf_device_map'):
             model.to(DEVICE)
             logger.info(f"Модель вручную перемещена на устройство: {DEVICE}")

        load_end_time = time.time()
        logger.info(f"LLM модель и токенизатор успешно загружены за {load_end_time - load_start_time:.2f} сек.")
        return model, tokenizer

    except ImportError as e:
        logger.critical(f"Ошибка импорта при загрузке LLM: {e}. Убедитесь, что все зависимости установлены.")
    except ValueError as e: # Ловим ошибку с токенами
         logger.critical(f"Ошибка конфигурации токенизатора: {e}")
    except Exception as e:
        logger.critical(f"Не удалось загрузить LLM модель или токенизатор: {e}", exc_info=True)

    # Очистка в случае ошибки
    unload_llm_model()
    return None, None


# --- Генерация текста с параметрами ---
def generate_with_params(
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    **kwargs # Для возможных дополнительных параметров generate()
) -> str:
    """
    Генерирует текст с использованием LLM, позволяя указать параметры генерации.

    Args:
        messages (List[Dict[str, str]]): Список сообщений для промпта.
        max_new_tokens (int): Макс. количество новых токенов.
        temperature (float): Температура для сэмплирования.
        top_p (float): Top-p (nucleus) сэмплирование.
        top_k (int): Top-k сэмплирование.
        repetition_penalty (float): Штраф за повторения.
        **kwargs: Дополнительные аргументы для model.generate().

    Returns:
        str: Сгенерированный ответ модели или строка с описанием ошибки.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.error("LLM Модель или токенизатор не загружены для генерации.")
        return "Ошибка: LLM не загружена."
    if tokenizer.pad_token_id is None:
        logger.error("tokenizer.pad_token_id не установлен! Генерация невозможна.")
        return "Ошибка: Не настроен pad_token_id."

    try:
        # Применяем шаблон чата
        # tokenize=False, т.к. токенизируем ниже перед generate
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Добавляет маркер assistant
        )
    except Exception as e:
        logger.error(f"Ошибка при применении шаблона чата: {e}", exc_info=True)
        return "Ошибка: Не удалось сформировать промпт."

    logger.debug(f"--- Финальный промпт для LLM (len={len(formatted_prompt)}) ---\n{formatted_prompt[:500]}...\n--------------------")

    generation_start_time = time.time()
    try:
        # Токенизация
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(DEVICE)
        input_length = inputs.input_ids.shape[1]

        # Проверка на пустой ввод после токенизации (маловероятно, но возможно)
        if input_length == 0:
            logger.warning("Промпт стал пустым после токенизации.")
            return "Ошибка: Пустой промпт."

        # Генерация с заданными параметрами
        with torch.no_grad():
             outputs = model.generate(
                 **inputs,
                 max_new_tokens=max_new_tokens,
                 do_sample=True, # Всегда используем сэмплирование для ролеплея
                 temperature=temperature,
                 top_p=top_p,
                 top_k=top_k,
                 repetition_penalty=repetition_penalty,
                 # Указываем терминальные токены Llama 3 и pad token
                 eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                 pad_token_id=tokenizer.pad_token_id,
                 **kwargs # Передаем доп. аргументы, если есть
             )

        # Декодирование только сгенерированной части
        generated_ids = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        generation_end_time = time.time()
        gen_time = generation_end_time - generation_start_time
        token_count = len(generated_ids)
        tokens_per_sec = token_count / gen_time if gen_time > 0 else 0
        logger.info(f"Генерация: {token_count} токен(ов) за {gen_time:.2f} сек ({tokens_per_sec:.2f} токен/сек).")
        logger.debug(f"--- Ответ LLM ---\n{generated_text}\n-----------------")

        return generated_text.strip()

    except Exception as e:
        logger.error(f"Ошибка при генерации текста: {e}", exc_info=True)
        if DEVICE == 'cuda':
            torch.cuda.empty_cache() # Попытка очистить память GPU
            gc.collect()
        return "Ошибка генерации ответа."


# --- Старая функция generate_text (закомментирована, т.к. заменена на generate_with_params) ---
# def generate_text(messages: List[Dict[str, str]]) -> str:
#     """(УСТАРЕЛО) Генерирует текст с параметрами по умолчанию."""
#     logger.warning("Вызов устаревшей функции generate_text. Используйте generate_with_params.")
#     # Вызов новой функции с параметрами из config (если они там еще есть)
#     from config import GENERATION_MAX_NEW_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P, GENERATION_TOP_K, GENERATION_REPETITION_PENALTY
#     return generate_with_params(
#         messages=messages,
#         max_new_tokens=GENERATION_MAX_NEW_TOKENS,
#         temperature=GENERATION_TEMPERATURE,
#         top_p=GENERATION_TOP_P,
#         top_k=GENERATION_TOP_K,
#         repetition_penalty=GENERATION_REPETITION_PENALTY
#     )


# --- Выгрузка Модели ---
def unload_llm_model():
    """Освобождает память, занимаемую LLM моделью и токенизатором."""
    global model, tokenizer
    if model is not None:
        logger.info("Выгрузка LLM модели...")
        del model
        model = None
    if tokenizer is not None:
        logger.info("Выгрузка токенизатора...")
        del tokenizer
        tokenizer = None

    if DEVICE == 'cuda':
        logger.info("Очистка кэша CUDA...")
        torch.cuda.empty_cache()

    gc.collect()
    logger.info("LLM Модель и токенизатор выгружены.")


# --- Локальный Тест ---
if __name__ == "__main__":
    print("Запуск локального теста llama_loader с generate_with_params...")
    test_model, test_tokenizer = load_llm_model()
    if test_model and test_tokenizer:
        print("-" * 20)
        test_messages: List[Message] = [
            {"role": "system", "content": "Ты - пират Капитан Крюк. Потерял руку в бою с крокодилом."},
            {"role": "user", "content": "Привет, Капитан! Как поживает твой крюк?"}
        ]
        print(f"Тестовые сообщения: {test_messages}")

        response = generate_with_params(
            messages=test_messages,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
        )
        print(f"Ответ LLM:\n{response}")
        print("-" * 20)
    else:
         print("Не удалось загрузить модель для теста.")

    unload_llm_model()
    print("Тест llama_loader завершен.")