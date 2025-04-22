# memory/short_term.py
from collections import deque
from typing import List, Dict, Optional, Deque

from config import STM_MAX_MESSAGES, logger

# Определяем формат сообщения, который будем использовать
# Совместим с ожидаемым форматом Llama 3 и transformers.apply_chat_template
Message = Dict[str, str] # Пример: {"role": "user", "content": "Привет"}

class ShortTermMemory:
    """
    Управляет кратковременной памятью диалога (последние N сообщений).
    Использует deque для автоматического удаления старых сообщений.
    Хранит сообщения в формате словарей {"role": "...", "content": "..."}.
    """
    def __init__(self, max_messages: int = STM_MAX_MESSAGES):
        """
        Инициализирует STM.

        Args:
            max_messages (int): Максимальное количество сообщений (не пар)
                                 для хранения в памяти.
        """
        if max_messages <= 0:
            logger.warning(f"Некорректное значение max_messages ({max_messages}). Установлено в 10.")
            max_messages = 10
        # Храним отдельные сообщения (user, assistant), а не пары
        self.max_messages = max_messages
        self.messages: Deque[Message] = deque(maxlen=max_messages)
        logger.debug(f"STM инициализирована с лимитом {max_messages} сообщений.")

    def add_message(self, role: str, content: str):
        """
        Добавляет одно сообщение в память.

        Args:
            role (str): Роль отправителя ("user" или "assistant").
            content (str): Текст сообщения.
        """
        if role not in ["user", "assistant"]:
            logger.warning(f"Некорректная роль сообщения '{role}' в STM. Сообщение проигнорировано.")
            return
        if not isinstance(content, str):
            logger.warning(f"Некорректный тип контента ({type(content)}) в STM. Ожидалась строка.")
            # Попытка преобразовать в строку, если возможно
            try:
                content = str(content)
            except Exception:
                 logger.error("Не удалось преобразовать контент в строку. Сообщение проигнорировано.")
                 return

        message: Message = {"role": role, "content": content.strip()}
        self.messages.append(message)
        logger.debug(f"Сообщение добавлено в STM: {message}")

    def add_message_pair(self, user_message: str, assistant_message: str):
        """
        Добавляет пару сообщений (пользователь и ассистент) в память.
        Удобный метод для добавления после обмена репликами.
        """
        self.add_message(role="user", content=user_message)
        self.add_message(role="assistant", content=assistant_message)

    def get_messages(self) -> List[Message]:
        """
        Возвращает список всех сообщений в текущей памяти (от старых к новым).
        """
        return list(self.messages)

    def load_messages(self, messages: List[Message]):
        """
        Загружает список сообщений в память, заменяя существующие.
        Обычно используется при загрузке состояния из БД.
        Сохраняет только последние `max_messages` сообщений.

        Args:
            messages (List[Message]): Список сообщений для загрузки.
        """
        if not isinstance(messages, list):
            logger.error(f"Ошибка загрузки STM: ожидался список, получен {type(messages)}.")
            self.messages.clear()
            return

        # Очищаем текущую память перед загрузкой
        self.messages.clear()

        # Берем только последние N сообщений, если загружаемый список больше лимита
        start_index = max(0, len(messages) - self.max_messages)
        messages_to_load = messages[start_index:]

        # Валидируем и добавляем сообщения
        valid_messages = []
        for msg in messages_to_load:
            if isinstance(msg, dict) and "role" in msg and "content" in msg \
               and msg["role"] in ["user", "assistant"] and isinstance(msg["content"], str):
                valid_messages.append(msg)
            else:
                logger.warning(f"Обнаружено некорректное сообщение при загрузке STM: {msg}. Пропущено.")

        self.messages.extend(valid_messages)
        logger.debug(f"STM загружена. Загружено {len(self.messages)} сообщений.")


    def clear(self):
        """Очищает память."""
        self.messages.clear()
        logger.debug("STM очищена.")

    def __len__(self) -> int:
        """Возвращает текущее количество сообщений в памяти."""
        return len(self.messages)

    def __str__(self) -> str:
        """Строковое представление для отладки."""
        return f"ShortTermMemory(count={len(self.messages)}, max={self.max_messages}, messages={list(self.messages)})"