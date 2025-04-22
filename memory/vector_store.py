# memory/vector_store.py
import chromadb
# from chromadb.config import Settings # Не используется явно, можно убрать
from chromadb.errors import IDAlreadyExistsError, CollectionNotFoundError, InvalidDimensionException # Добавляем InvalidDimensionException
import uuid
import datetime
from typing import List, Tuple, Optional, Dict # Добавили Optional, Dict

from config import VECTOR_STORE_PATH, logger
# Импортируем саму модель эмбеддингов, чтобы проверить её размерность
from memory.embeddings import get_embedding, embedding_model

# --- Инициализация клиента ChromaDB ---
try:
    # Используем настройки по умолчанию, которые включают DuckDB + Parquet
    # Увеличим таймаут для избежания проблем при высокой нагрузке
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_STORE_PATH
        # settings=Settings(allow_reset=True) # Можно разрешить сброс, если нужно
    )
    # Проверим работоспособность клиента (например, запросив heartbeat)
    chroma_client.heartbeat() # Выбросит исключение, если клиент не работает
    logger.info(f"ChromaDB клиент успешно инициализирован. Путь: {VECTOR_STORE_PATH}")

    # --- Получаем размерность эмбеддингов ---
    # Это нужно для более надежного создания коллекций
    if embedding_model:
         # Получаем размерность из загруженной модели
         EMBEDDING_DIMENSION = embedding_model.get_sentence_embedding_dimension()
         logger.info(f"Определена размерность эмбеддингов: {EMBEDDING_DIMENSION}")
    else:
         # Если модель не загружена, используем стандартное значение или None
         # Это может вызвать проблемы позже, если модель будет иметь другую размерность
         EMBEDDING_DIMENSION = None # Или стандартное значение, например 384 для all-MiniLM-L6-v2
         logger.warning("Модель эмбеддингов не загружена при инициализации VSManager. Размерность не определена.")

except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать ChromaDB клиент в {VECTOR_STORE_PATH}: {e}", exc_info=True)
    chroma_client = None
    EMBEDDING_DIMENSION = None # Явно ставим None

class VectorStoreManager:
    """
    Предоставляет интерфейс для взаимодействия с векторной базой данных ChromaDB.
    """
    def __init__(self):
        if chroma_client is None:
            # Исключение выбросится при инициализации chroma_client
            raise RuntimeError("ChromaDB клиент не был инициализирован.")
        self.client = chroma_client
        # Сохраняем размерность для использования при создании коллекций
        self.embedding_dimension = EMBEDDING_DIMENSION

    def _get_or_create_collection(self, collection_name: str) -> Optional[chromadb.Collection]:
        """Получает или создает коллекцию в ChromaDB."""
        if not collection_name or not isinstance(collection_name, str):
             logger.error("Некорректное имя коллекции передано в _get_or_create_collection.")
             return None
        try:
            # Передаем размерность эмбеддингов, если она известна
            # Это помогает ChromaDB оптимизировать хранение и поиск
            metadata = {"hnsw:space": "cosine"} # Косинусное расстояние - стандарт для норм. эмбеддингов
            if self.embedding_dimension:
                 # Указываем размерность при создании (может быть проигнорировано, если коллекция уже существует)
                 # В новых версиях ChromaDB это может быть не обязательно явно указывать
                 pass # В chromadb >= 0.4.15 размерность часто определяется автоматически

            # Пытаемся получить коллекцию
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"Коллекция '{collection_name}' найдена.")
                return collection
            except Exception as get_e: # Ловим ошибку получения (может быть не только CollectionNotFoundError)
                logger.debug(f"Коллекция '{collection_name}' не найдена, попытка создания...")
                # Если не найдена, создаем
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata=metadata
                    # embedding_function=None # Не используем встроенную функцию эмбеддинга
                )
                logger.info(f"Коллекция '{collection_name}' успешно создана.")
                return collection

        except Exception as e:
            logger.error(f"Ошибка при получении/создании коллекции '{collection_name}': {e}", exc_info=True)
            return None

    def add_memory(self, collection_name: str, memory_text: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Добавляет текстовое воспоминание (факт) в указанную коллекцию.

        Args:
            collection_name (str): Имя коллекции.
            memory_text (str): Текст воспоминания/факта.
            metadata (Optional[Dict]): Дополнительные метаданные.

        Returns:
            Optional[str]: Уникальный ID добавленного документа или None в случае ошибки.
        """
        collection = self._get_or_create_collection(collection_name)
        if not collection:
            logger.error(f"Не удалось получить/создать коллекцию '{collection_name}' для добавления памяти.")
            return None

        embedding = get_embedding(memory_text)
        if not embedding:
            logger.error(f"Не удалось сгенерировать эмбеддинг для добавления в '{collection_name}'. Текст: '{memory_text[:50]}...'")
            return None

        # Проверка размерности эмбеддинга (если она известна)
        if self.embedding_dimension and len(embedding) != self.embedding_dimension:
             logger.error(f"Размерность сгенерированного эмбеддинга ({len(embedding)}) не совпадает с ожидаемой ({self.embedding_dimension}) для коллекции '{collection_name}'. Документ не добавлен.")
             return None

        doc_id = str(uuid.uuid4())
        doc_metadata = metadata if metadata is not None else {}
        if 'timestamp' not in doc_metadata:
            doc_metadata['timestamp'] = datetime.datetime.utcnow().isoformat()

        try:
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[memory_text], # Сохраняем сам текст
                metadatas=[doc_metadata]
            )
            # logger.info(f"Воспоминание добавлено в '{collection_name}' (ID: {doc_id})") # Логируется в LongTermMemory
            return doc_id
        except IDAlreadyExistsError:
             logger.warning(f"Сгенерирован уже существующий ID {doc_id} для '{collection_name}'.")
             return None # Не повторяем попытку, т.к. это баг или редчайшее совпадение
        except InvalidDimensionException as e:
             logger.error(f"Ошибка размерности при добавлении в коллекцию '{collection_name}': {e}. Убедитесь, что все эмбеддинги имеют одинаковую размерность ({self.embedding_dimension}).", exc_info=True)
             return None
        except Exception as e:
             logger.error(f"Ошибка при добавлении воспоминания в коллекцию '{collection_name}': {e}", exc_info=True)
             return None

    def search_relevant_memories(self, collection_name: str, query_text: str, n_results: int = 3) -> List[Tuple[str, float]]:
        """
        Ищет наиболее релевантные воспоминания в коллекции.

        Args:
            collection_name (str): Имя коллекции.
            query_text (str): Текст запроса.
            n_results (int): Количество результатов.

        Returns:
            List[Tuple[str, float]]: Список кортежей (текст_воспоминания, расстояние).
        """
        try:
            collection = self.client.get_collection(name=collection_name)
        except CollectionNotFoundError:
             logger.debug(f"Коллекция '{collection_name}' не найдена для поиска.")
             return []
        except Exception as e:
             logger.error(f"Ошибка при получении коллекции '{collection_name}' для поиска: {e}", exc_info=True)
             return []

        query_embedding = get_embedding(query_text)
        if not query_embedding:
            logger.error(f"Не удалось сгенерировать эмбеддинг для запроса поиска в '{collection_name}'.")
            return []

        # Проверка размерности запроса
        if self.embedding_dimension and len(query_embedding) != self.embedding_dimension:
             logger.error(f"Размерность эмбеддинга запроса ({len(query_embedding)}) не совпадает с ожидаемой ({self.embedding_dimension}) для коллекции '{collection_name}'. Поиск отменен.")
             return []

        try:
            count = collection.count()
            if count == 0:
                # logger.debug(f"Коллекция '{collection_name}' пуста.") # Не логируем, т.к. это частый случай
                return []

            actual_n_results = max(1, min(n_results, count)) # Запрашиваем хотя бы 1, если есть что искать

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_n_results,
                include=['documents', 'distances']
            )

            # Обработка результатов (как раньше)
            retrieved_memories = []
            if results and results.get('documents') and results.get('distances') and len(results['documents']) > 0:
                 docs = results['documents'][0]
                 dists = results['distances'][0]
                 if len(docs) == len(dists):
                      for doc, dist in zip(docs, dists):
                           if doc is not None and dist is not None: # Доп. проверка на None
                                retrieved_memories.append((doc, float(dist)))
                      # logger.debug(f"Найдено {len(retrieved_memories)} релевантных воспоминаний в '{collection_name}'.") # Логируется в LongTermMemory
                      return retrieved_memories

            logger.debug(f"Поиск в '{collection_name}' не дал валидных результатов.")
            return []

        except InvalidDimensionException as e:
             logger.error(f"Ошибка размерности при поиске в коллекции '{collection_name}': {e}.", exc_info=True)
             return []
        except Exception as e:
            logger.error(f"Ошибка при поиске в коллекции '{collection_name}': {e}", exc_info=True)
            return []

    def delete_collection(self, collection_name: str):
        """Удаляет коллекцию."""
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Коллекция '{collection_name}' успешно удалена.")
        except CollectionNotFoundError:
             logger.warning(f"Попытка удалить несуществующую коллекцию: '{collection_name}'.")
        except Exception as e:
             # Используем warning, так как это может быть не критично, если коллекция уже удалена
             logger.warning(f"Ошибка при удалении коллекции '{collection_name}': {e}", exc_info=True)

    def list_collections(self) -> List[str]:
        """Возвращает список имен всех коллекций."""
        try:
             collections = self.client.list_collections()
             return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Ошибка при получении списка коллекций: {e}", exc_info=True)
            return []