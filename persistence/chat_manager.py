# persistence/chat_manager.py (Refactored)
import datetime
import hashlib
from typing import Dict, Optional

from config import logger # Убираем MAX_ACTIVE_CHATS
from persistence.chat_state import ChatState
from persistence.database import save_chat_state, load_chat_state, delete_chat_state
from memory.vector_store import VectorStoreManager

class ChatManager:
    """
    Управляет состояниями чатов: загружает по требованию,
    обрабатывает выбор и сброс сценариев.
    """
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.active_chats: Dict[int, ChatState] = {} # Только активные в памяти
        self.vector_store_manager = vector_store_manager
        logger.info("ChatManager инициализирован с VectorStoreManager.")

    def _generate_collection_name(self, chat_id: int, user_id: int) -> str:
         """Генерирует имя коллекции LTM для пары чат-пользователь."""
         # Оставляем привязку LTM к чату/пользователю, а не к сценарию
         base_name = f"chat_{chat_id}_user_{user_id}"
         return hashlib.sha1(base_name.encode()).hexdigest()[:16] # Короткий хэш

    async def get_chat_state(self, chat_id: int, user_id: int) -> ChatState:
        """
        Возвращает ChatState для данного chat_id.
        Загружает из памяти, БД или создает новый.
        Гарантирует наличие LTM коллекции.
        """
        if chat_id in self.active_chats:
            state = self.active_chats[chat_id]
            # Обновляем время только при реальном взаимодействии (в handle_message)
            # state.update_interaction_time()
            logger.debug(f"Чат {chat_id} найден в активной памяти.")
            return state

        logger.debug(f"Чат {chat_id} не найден в памяти, попытка загрузить из БД...")
        # load_chat_state должна быть асинхронной или выполняться в потоке
        # Пока считаем её синхронной для простоты, но в реальном приложении лучше to_thread
        state = load_chat_state(chat_id) # TODO: Сделать асинхронной?

        if state:
            logger.info(f"Чат {chat_id} загружен из БД.")
            # Проверка LTM ID и коллекции (как раньше)
            if not state.ltm_collection_id:
                state.ltm_collection_id = self._generate_collection_name(chat_id, user_id)
                logger.info(f"Установлен LTM collection ID '{state.ltm_collection_id}' для существующего чата {chat_id}.")
                save_chat_state(state) # TODO: Сделать асинхронной?
            try:
                # Гарантируем существование коллекции
                await asyncio.to_thread(self.vector_store_manager._get_or_create_collection, state.ltm_collection_id)
            except Exception as e:
                 logger.error(f"Не удалось проверить/создать LTM коллекцию {state.ltm_collection_id} для чата {chat_id}: {e}")
                 # Продолжаем работу, но LTM может быть недоступна

            self.active_chats[chat_id] = state # Добавляем в активную память
            return state
        else:
            # Создаем новый state
            logger.info(f"Создание нового состояния для чата {chat_id}.")
            ltm_id = self._generate_collection_name(chat_id, user_id)
            try:
                await asyncio.to_thread(self.vector_store_manager._get_or_create_collection, ltm_id)
                logger.info(f"Создана LTM коллекция '{ltm_id}' для нового чата {chat_id}.")
            except Exception as e:
                 logger.error(f"Не удалось создать LTM коллекцию {ltm_id} для нового чата {chat_id}: {e}")
                 ltm_id = None # Не сохраняем ID, если не удалось создать

            new_state = ChatState(
                chat_id=chat_id,
                user_id=user_id,
                ltm_collection_id=ltm_id,
                scenario_id=None, # Явно указываем, что сценарий не выбран
                system_prompt=None # И динамический промпт пуст
            )
            self.active_chats[chat_id] = new_state
            save_chat_state(new_state) # TODO: Сделать асинхронной?
            return new_state

    async def get_existing_chat_state(self, chat_id: int) -> Optional[ChatState]:
        """
        Возвращает ChatState, если он есть в памяти или БД. НЕ создает новый.
        """
        if chat_id in self.active_chats:
            return self.active_chats[chat_id]

        # TODO: Сделать load_chat_state асинхронной?
        return load_chat_state(chat_id)

    async def set_scenario(self, chat_id: int, user_id: int, scenario_id: int, initial_prompt: str) -> ChatState:
        """
        Устанавливает или переключает сценарий для чата.
        Сбрасывает STM и system_prompt при смене сценария, но НЕ LTM.
        """
        state = await self.get_chat_state(chat_id, user_id) # Получаем или создаем state

        if state.scenario_id != scenario_id:
            logger.info(f"Смена сценария для чата {chat_id} с {state.scenario_id} на {scenario_id}. Сброс STM и системного промпта.")
            state.scenario_id = scenario_id
            state.system_prompt = initial_prompt # Инициализируем динамический промпт начальным
            state.clear_stm() # Очищаем кратковременную память
            state.update_interaction_time()
            # LTM НЕ ТРОГАЕМ при смене сценария
            save_chat_state(state) # TODO: Сделать асинхронной?
        else:
            # Пользователь выбрал тот же сценарий, просто продолжаем
            logger.info(f"Продолжение сценария {scenario_id} для чата {chat_id}.")
            # Обновим время взаимодействия на всякий случай
            state.update_interaction_time()

        return state

    async def delete_chat_session(self, chat_id: int) -> bool:
        """
        Сбрасывает/удаляет сессию чата: удаляет из памяти, БД и удаляет LTM коллекцию.
        Возвращает True, если сессия найдена и удалена, иначе False.
        """
        logger.warning(f"Запрос на сброс/удаление сессии для чата {chat_id}.")
        state_to_delete: Optional[ChatState] = None
        existed = False

        if chat_id in self.active_chats:
            state_to_delete = self.active_chats.pop(chat_id)
            existed = True
            logger.debug(f"Чат {chat_id} удален из активной памяти.")
        else:
            # Загружаем из БД, чтобы получить ltm_id и убедиться, что запись есть
            state_to_delete = load_chat_state(chat_id) # TODO: Сделать асинхронной?
            if state_to_delete:
                existed = True

        if not existed:
            logger.warning(f"Попытка сбросить/удалить несуществующую сессию чата {chat_id}.")
            return False

        # Удаление LTM коллекции (в потоке, т.к. может быть I/O)
        if state_to_delete and state_to_delete.ltm_collection_id:
            ltm_id_to_delete = state_to_delete.ltm_collection_id
            logger.info(f"Запланировано удаление LTM коллекции '{ltm_id_to_delete}' для чата {chat_id}.")
            try:
                # Запускаем удаление асинхронно
                await asyncio.to_thread(self.vector_store_manager.delete_collection, ltm_id_to_delete)
                logger.info(f"LTM коллекция '{ltm_id_to_delete}' для чата {chat_id} успешно удалена.")
            except Exception as e:
                 logger.error(f"Ошибка при удалении LTM коллекции {ltm_id_to_delete} для чата {chat_id}: {e}")
        elif state_to_delete:
            logger.warning(f"У чата {chat_id} не было LTM collection ID при удалении.")

        # Удаляем запись из SQLite
        try:
             delete_chat_state(chat_id) # TODO: Сделать асинхронной?
             logger.info(f"Запись для чата {chat_id} удалена из SQLite.")
        except Exception as e:
             logger.error(f"Ошибка при удалении записи чата {chat_id} из SQLite: {e}")
             # Даже если не удалось удалить из SQLite, LTM уже удалена (или попытка была)

        logger.info(f"Сессия чата {chat_id} полностью сброшена/удалена.")
        return True

    async def save_all_active_chats(self):
        """Сохраняет все активные чаты из памяти в БД."""
        if not self.active_chats:
            return
        logger.info(f"Сохранение состояний {len(self.active_chats)} активных чатов в БД...")
        saved_count = 0
        # Преобразуем в список, чтобы избежать проблем при изменении словаря во время итерации (хотя здесь это маловероятно)
        active_chat_items = list(self.active_chats.items())
        for chat_id, state in active_chat_items:
            try:
                # TODO: Сделать save_chat_state асинхронной?
                save_chat_state(state)
                saved_count += 1
            except Exception as e:
                 logger.error(f"Не удалось сохранить состояние чата {chat_id} при массовом сохранении: {e}")
        logger.info(f"Сохранено {saved_count} из {len(active_chat_items)} активных чатов.")

    # Методы _ensure_capacity и unload_chat удалены, так как лимита активных чатов больше нет.
    # Старый метод set_system_prompt удален, заменен на set_scenario.