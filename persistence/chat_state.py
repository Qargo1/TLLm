# persistence/chat_state.py
from dataclasses import dataclass, field
import datetime
from typing import Optional, List

# Импортируем класс STM
from memory.short_term import ShortTermMemory, Message # Убедимся, что Message импортируется
from config import STM_MAX_MESSAGES # Импортируем лимит для инициализации

@dataclass
class ChatState:
    """Хранит состояние одного чата."""
    chat_id: int
    user_id: int
    # --- Добавлено поле scenario_id ---
    scenario_id: Optional[int] = None # ID текущего выбранного сценария
    # --- Динамический системный промпт ---
    # Начинается с initial_system_prompt сценария, потом обновляется рефлексией
    system_prompt: Optional[str] = None
    # --- Память ---
    short_term_memory: ShortTermMemory = field(default_factory=lambda: ShortTermMemory(max_messages=STM_MAX_MESSAGES))
    ltm_collection_id: Optional[str] = None # ID коллекции в ChromaDB (привязан к chat_id/user_id)
    # --- Метаданные ---
    last_interaction_time: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    message_counter: int = 0 # Счетчик сообщений для запуска рефлексии

    def update_interaction_time(self):
        """Обновляет время последнего взаимодействия."""
        self.last_interaction_time = datetime.datetime.utcnow()

    def increment_message_counter(self):
        """Увеличивает счетчик сообщений."""
        self.message_counter += 1

    def add_stm_entry(self, user_message: str, bot_message: str):
        """Добавляет пару сообщений в STM и обновляет метаданные."""
        self.short_term_memory.add_message_pair(user_message, bot_message)
        self.update_interaction_time()
        # Увеличиваем счетчик на 2 (user + bot сообщения)
        self.increment_message_counter()
        self.increment_message_counter()

    def get_stm_messages(self) -> List[Message]:
        """Возвращает список сообщений из STM."""
        return self.short_term_memory.get_messages()

    def clear_stm(self):
        """Очищает STM и сбрасывает счетчик сообщений."""
        self.short_term_memory.clear()
        self.message_counter = 0 # Сбрасываем счетчик при очистке STM
        self.update_interaction_time()

    # Добавляем метод для обновления динамического промпта
    def update_dynamic_prompt(self, new_prompt: str):
        """Обновляет динамический системный промпт."""
        self.system_prompt = new_prompt
        # Можно добавить логирование изменения