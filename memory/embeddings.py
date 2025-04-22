# memory/embeddings.py
from sentence_transformers import SentenceTransformer
from typing import Optional, List # Добавляем типизацию
import numpy as np # Добавляем numpy для работы с результатом encode

from config import EMBEDDING_MODEL_NAME, logger, DEVICE
import time
import gc
import torch

# --- Глобальная переменная для модели эмбеддингов ---
embedding_model: Optional[SentenceTransformer] = None

# --- Загрузка Модели ---
def load_embedding_model() -> Optional[SentenceTransformer]:
    """
    Загружает модель Sentence Transformer.
    Возвращает загруженную модель или None в случае ошибки.
    """
    global embedding_model
    if embedding_model is not None:
        logger.debug("Модель эмбеддингов уже загружена.")
        return embedding_model

    logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
    load_start_time = time.time()
    try:
        # Явно указываем устройство из config
        loaded_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        # Проверка, что модель действительно загрузилась (иногда может вернуть None или сломаться позже)
        if loaded_model is None or not hasattr(loaded_model, 'encode'):
             raise RuntimeError("SentenceTransformer вернул некорректный объект модели.")

        embedding_model = loaded_model # Присваиваем глобальной переменной только после успеха
        load_end_time = time.time()
        logger.info(f"Модель эмбеддингов '{EMBEDDING_MODEL_NAME}' загружена за {load_end_time - load_start_time:.2f} сек на устройстве {embedding_model.device}.")
        return embedding_model
    except Exception as e:
        logger.error(f"Не удалось загрузить модель эмбеддингов '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
        embedding_model = None # Убедимся, что None в случае ошибки
        return None

# --- Генерация Эмбеддинга (один текст) ---
def get_embedding(text: str) -> Optional[List[float]]:
    """
    Генерирует нормализованный эмбеддинг для одного текста.

    Args:
        text (str): Входной текст.

    Returns:
        Optional[List[float]]: Список float чисел (эмбеддинг) или None при ошибке.
    """
    global embedding_model
    if embedding_model is None:
        logger.error("Модель эмбеддингов не загружена для get_embedding.")
        return None
    if not text or not isinstance(text, str):
        # Не логируем как warning, т.к. пустой текст может приходить штатно
        logger.debug("Попытка сгенерировать эмбеддинг для пустого или некорректного текста.")
        return None

    try:
        # normalize_embeddings=True важно для косинусного сходства в ChromaDB
        embedding: np.ndarray = embedding_model.encode(
            text,
            convert_to_numpy=True, # Получаем numpy массив
            normalize_embeddings=True
        )
        # Проверяем, что результат не пустой и имеет ожидаемую структуру
        if embedding is None or embedding.ndim != 1:
             logger.error(f"Модель эмбеддингов вернула некорректный результат для текста: '{text[:50]}...'")
             return None
        # Конвертируем numpy array в стандартный Python list[float]
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Ошибка при генерации эмбеддинга для текста '{text[:50]}...': {e}", exc_info=True)
        return None

# --- Генерация Эмбеддингов (батч) ---
def get_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Генерирует нормализованные эмбеддинги для списка текстов (батчинг).

    Args:
        texts (List[str]): Список входных текстов.

    Returns:
        Optional[List[List[float]]]: Список эмбеддингов или None при ошибке.
    """
    global embedding_model
    if embedding_model is None:
        logger.error("Модель эмбеддингов не загружена для get_embeddings.")
        return None
    # Проверяем корректность входного списка
    if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        logger.warning("Попытка сгенерировать эмбеддинги для некорректного списка текстов.")
        return None

    try:
        # Используем encode батчем
        embeddings: np.ndarray = embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
            # batch_size= можно указать, если нужно управлять размером батча
        )
        # Проверка результата
        if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            logger.error("Модель эмбеддингов вернула некорректный результат для батча.")
            return None
        # Конвертируем в список списков float
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Ошибка при генерации эмбеддингов батчем: {e}", exc_info=True)
        return None

# --- Выгрузка Модели ---
def unload_embedding_model():
    """Выгружает модель эмбеддингов и очищает память."""
    global embedding_model
    if embedding_model is not None:
        logger.info("Выгрузка модели эмбеддингов...")
        # Запоминаем устройство перед удалением объекта
        model_device_str = str(embedding_model.device)
        # Удаляем объект модели
        del embedding_model
        embedding_model = None # Устанавливаем в None
        # Очищаем кэш CUDA, если модель была на GPU
        if model_device_str.startswith('cuda'):
             logger.info("Очистка кэша CUDA после выгрузки модели эмбеддингов...")
             torch.cuda.empty_cache()
        # Запускаем сборщик мусора Python
        gc.collect()
        logger.info("Модель эмбеддингов выгружена.")
    else:
        logger.debug("Модель эмбеддингов не была загружена, выгрузка не требуется.")

# --- Локальный Тест ---
if __name__ == "__main__":
    print("--- Запуск локального теста embeddings ---")
    loaded_model = load_embedding_model()
    if loaded_model:
        print(f"Модель '{EMBEDDING_MODEL_NAME}' загружена на устройство: {loaded_model.device}")
        # Получаем размерность из модели
        try:
            dimension = loaded_model.get_sentence_embedding_dimension()
            print(f"Ожидаемая размерность эмбеддинга: {dimension}")
        except Exception as e:
            print(f"Не удалось получить размерность: {e}")
            dimension = None

        print("\nТест 1: Один текст")
        test_text = "Это пример текста для проверки эмбеддинга."
        emb = get_embedding(test_text)
        if emb:
            print(f"  Текст: '{test_text}'")
            print(f"  Получен эмбеддинг размерности: {len(emb)}")
            if dimension and len(emb) != dimension:
                print(f"  ВНИМАНИЕ: Размерность ({len(emb)}) не совпадает с ожидаемой ({dimension})!")
            print(f"  Первые 5 значений: {emb[:5]}")
        else:
            print(f"  Не удалось сгенерировать эмбеддинг для: '{test_text}'")

        print("\nТест 2: Батч текстов")
        test_texts = ["Первый текст в батче.", "Второй, немного другой текст.", ""] # Добавим пустой для теста
        embs = get_embeddings(test_texts)
        if embs:
            print(f"  Тексты: {test_texts}")
            print(f"  Получено эмбеддингов: {len(embs)}")
            if len(embs) > 0:
                print(f"  Размерность первого: {len(embs[0]) if embs[0] else 'N/A'}")
                # Пустой текст должен вернуть эмбеддинг (обычно нулевой или близкий к нему после нормализации)
                print(f"  Размерность для пустого текста ('{test_texts[2]}'): {len(embs[2]) if len(embs) > 2 and embs[2] is not None else 'N/A'}")
                if dimension and embs[0] and len(embs[0]) != dimension:
                     print(f"  ВНИМАНИЕ: Размерность ({len(embs[0])}) не совпадает с ожидаемой ({dimension})!")
            else:
                 print("  Батч вернул пустой список.")
        else:
            print(f"  Не удалось сгенерировать эмбеддинги для батча: {test_texts}")

        print("\nТест 3: Пустой текст отдельно")
        empty_emb = get_embedding("")
        if empty_emb is not None: # Пустой текст может вернуть эмбеддинг
            print(f"  Эмбеддинг для пустой строки ('') получен, размерность: {len(empty_emb)}")
        else:
            # Или может вернуть None, в зависимости от модели/обработки
            print(f"  get_embedding для пустой строки ('') вернул None (ожидаемо).")

    else:
        print(f"Не удалось загрузить модель '{EMBEDDING_MODEL_NAME}' для теста.")

    unload_embedding_model()
    print("--- Тест embeddings завершен ---")