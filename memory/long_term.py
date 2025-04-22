# memory/long_term.py
from typing import Optional, List # Убрали Tuple, т.к. возвращаем List[str]
import hashlib

from config import logger, LTM_RETRIEVAL_COUNT
from memory.vector_store import VectorStoreManager

class LongTermMemory:
    """
    Управляет долговременной памятью для одного чата, используя VectorStoreManager.
    """
    def __init__(self, chat_id: int, user_id: int, vector_store: VectorStoreManager):
        """
        Инициализирует LTM для конкретного чата.

        Args:
            chat_id (int): ID чата.
            user_id (int): ID пользователя.
            vector_store (VectorStoreManager): Экземпляр менеджера векторной БД.

        Raises:
            RuntimeError: Если не удалось получить/создать коллекцию в VectorStore.
        """
        self.chat_id = chat_id
        self.user_id = user_id
        self.vector_store = vector_store

        # Генерируем имя коллекции
        base_name = f"chat_{chat_id}_user_{user_id}"
        # Используем sha256 для большей уникальности и стандартности, но можно и sha1
        self.collection_name = hashlib.sha256(base_name.encode()).hexdigest()[:24] # Немного длиннее
        logger.debug(f"LTM для чата {chat_id}: инициализация с collection_name='{self.collection_name}'")

        # Пытаемся получить/создать коллекцию при инициализации
        # Метод _get_or_create_collection возвращает None в случае ошибки
        collection = self.vector_store._get_or_create_collection(self.collection_name)
        if collection is None:
            # Если не удалось создать/получить коллекцию, LTM не сможет работать.
            # Лучше выбросить исключение, чтобы это было обработано выше.
            msg = f"Не удалось инициализировать LTM для чата {chat_id}: не удалось получить/создать коллекцию '{self.collection_name}'"
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.debug(f"LTM для чата {chat_id}: коллекция '{self.collection_name}' подтверждена.")


    def add_memory(self, memory_text: str) -> bool:
        """
        Добавляет воспоминание (обычно факт, извлеченный рефлексией) в LTM этого чата.

        Args:
            memory_text (str): Текст воспоминания/факта.

        Returns:
            bool: True, если добавление успешно, иначе False.
        """
        if not memory_text or not isinstance(memory_text, str):
            logger.warning(f"Попытка добавить пустое или некорректное воспоминание в LTM чата {self.chat_id}")
            return False

        # Вызов VectorStoreManager для добавления
        # Предполагается, что VSM обрабатывает ошибки и возвращает ID или None
        doc_id = self.vector_store.add_memory(self.collection_name, memory_text)
        if doc_id:
            logger.info(f"Воспоминание добавлено в LTM чата {self.chat_id} (Коллекция: {self.collection_name})")
            return True
        else:
            logger.error(f"Не удалось добавить воспоминание в LTM чата {self.chat_id} (Коллекция: {self.collection_name})")
            return False

    def retrieve_relevant_memories(self, query_text: str, n: int = LTM_RETRIEVAL_COUNT) -> List[str]:
        """
        Извлекает n наиболее релевантных воспоминаний (фактов) из LTM по запросу.

        Args:
            query_text (str): Текст запроса (например, последнее сообщение пользователя).
            n (int): Максимальное количество воспоминаний для извлечения.

        Returns:
            List[str]: Список текстов релевантных воспоминаний.
        """
        if not query_text or not isinstance(query_text, str):
             logger.warning(f"Пустой или некорректный запрос для поиска в LTM чата {self.chat_id}")
             return []

        # Вызов VectorStoreManager для поиска
        # Он возвращает список кортежей (текст, расстояние) или пустой список
        results = self.vector_store.search_relevant_memories(
            collection_name=self.collection_name,
            query_text=query_text,
            n_results=n
        )

        memory_texts = [text for text, dist in results] # Извлекаем только тексты
        if memory_texts:
             logger.info(f"Извлечено {len(memory_texts)} релевантных факт(ов) из LTM чата {self.chat_id}.")
             logger.debug(f"Факты LTM для чата {self.chat_id}: {memory_texts}")
        else:
             logger.debug(f"Релевантных фактов в LTM для чата {self.chat_id} по запросу не найдено.")

        return memory_texts

    def delete_memory_store(self):
        """Полностью удаляет LTM хранилище (коллекцию) для этого чата."""
        logger.warning(f"Удаление LTM коллекции '{self.collection_name}' для чата {self.chat_id}.")
        # Вызов VectorStoreManager для удаления
        self.vector_store.delete_collection(self.collection_name)
        # Ошибки логируются внутри delete_collection

    def get_collection_name(self) -> str:
         """Возвращает имя коллекции, используемой этим экземпляром LTM."""
         return self.collection_name