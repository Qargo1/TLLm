# persistence/database.py
import sqlite3
import json
import datetime
from typing import Optional, List

from config import SQLITE_DB_NAME, logger, STM_MAX_MESSAGES
from persistence.chat_state import ChatState # Импортируем обновленный ChatState
from memory.short_term import ShortTermMemory, Message

# --- Инициализация базы данных ---
def initialize_database():
    """Создает/обновляет таблицу для хранения состояний чатов."""
    try:
        # Используем context manager для автоматического commit/rollback
        with sqlite3.connect(SQLITE_DB_NAME, timeout=10) as conn: # Увеличим таймаут
            cursor = conn.cursor()
            # Создаем таблицу, если не существует
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_states (
                    chat_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    scenario_id INTEGER,          -- ID выбранного сценария
                    system_prompt TEXT,           -- Динамический системный промпт
                    short_term_memory TEXT,       -- STM как JSON
                    ltm_collection_id TEXT,       -- ID LTM коллекции
                    last_interaction_time TEXT NOT NULL, -- ISO формат
                    message_counter INTEGER DEFAULT 0 -- Счетчик сообщений
                )
            """)

            # --- Проверка и добавление новых колонок (простая миграция) ---
            cursor.execute("PRAGMA table_info(chat_states)")
            columns = [info[1] for info in cursor.fetchall()]

            if 'scenario_id' not in columns:
                logger.warning("Обнаружено отсутствие колонки 'scenario_id'. Добавляю...")
                cursor.execute("ALTER TABLE chat_states ADD COLUMN scenario_id INTEGER")
                logger.info("Колонка 'scenario_id' добавлена.")

            if 'message_counter' not in columns:
                logger.warning("Обнаружено отсутствие колонки 'message_counter'. Добавляю...")
                cursor.execute("ALTER TABLE chat_states ADD COLUMN message_counter INTEGER DEFAULT 0")
                logger.info("Колонка 'message_counter' добавлена.")
            # --- Конец миграции ---

            conn.commit() # Явный коммит после ALTER TABLE
            logger.info(f"База данных '{SQLITE_DB_NAME}' инициализирована/обновлена успешно.")

    except sqlite3.Error as e:
        logger.critical(f"Критическая ошибка при инициализации/обновлении базы данных SQLite: {e}", exc_info=True)
        raise # Передаем исключение дальше

# --- Сохранение и загрузка ---
# TODO: Рассмотреть использование асинхронной библиотеки для SQLite (напр., aiosqlite)
#       или выполнять эти операции в потоке через asyncio.to_thread

def save_chat_state(chat_state: ChatState):
    """Сохраняет или обновляет состояние чата в базе данных."""
    if not isinstance(chat_state, ChatState):
        logger.error("Попытка сохранить не ChatState объект.")
        return

    # Проверка STM (как раньше)
    if not isinstance(chat_state.short_term_memory, ShortTermMemory):
        logger.error(f"Некорректный тип STM в ChatState для чата {chat_state.chat_id}.")
        return

    query = """
        INSERT OR REPLACE INTO chat_states (
            chat_id, user_id, scenario_id, system_prompt, short_term_memory,
            ltm_collection_id, last_interaction_time, message_counter
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        # Сериализация STM
        stm_json = json.dumps(chat_state.short_term_memory.get_messages())
        last_interaction_iso = chat_state.last_interaction_time.isoformat()

        params = (
            chat_state.chat_id,
            chat_state.user_id,
            chat_state.scenario_id, # Добавлено
            chat_state.system_prompt,
            stm_json,
            chat_state.ltm_collection_id,
            last_interaction_iso,
            chat_state.message_counter # Добавлено
        )

        with sqlite3.connect(SQLITE_DB_NAME, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
        logger.debug(f"Состояние чата {chat_state.chat_id} сохранено в БД.")

    except sqlite3.Error as e:
        logger.error(f"Ошибка SQLite при сохранении состояния чата {chat_state.chat_id}: {e}", exc_info=True)
    except json.JSONDecodeError as e:
         logger.error(f"Ошибка сериализации STM для чата {chat_state.chat_id}: {e}", exc_info=True)
    except Exception as e:
         logger.error(f"Непредвиденная ошибка при сохранении чата {chat_state.chat_id}: {e}", exc_info=True)

def load_chat_state(chat_id: int) -> Optional[ChatState]:
    """Загружает состояние чата из базы данных."""
    query = "SELECT * FROM chat_states WHERE chat_id = ?"
    try:
        with sqlite3.connect(SQLITE_DB_NAME, timeout=10) as conn:
            # Устанавливаем row_factory для получения словаря (удобнее)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (chat_id,))
            row = cursor.fetchone()

            if row:
                # Доступ к полям по имени благодаря row_factory
                try:
                    stm_json = row['short_term_memory']
                    stm_messages: List[Message] = []
                    if stm_json:
                        loaded_data = json.loads(stm_json)
                        if isinstance(loaded_data, list):
                            # Валидация (как раньше)
                            validated_messages = [
                                item for item in loaded_data
                                if isinstance(item, dict) and "role" in item and "content" in item
                            ]
                            stm_messages = validated_messages
                        else:
                             logger.warning(f"Некорректный тип STM из БД для чата {chat_id}.")
                    stm_instance = ShortTermMemory(max_messages=STM_MAX_MESSAGES)
                    stm_instance.load_messages(stm_messages)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Ошибка десериализации/доступа к STM для чата {chat_id}: {e}. Загружена пустая память.")
                    stm_instance = ShortTermMemory(max_messages=STM_MAX_MESSAGES)

                try:
                    last_interaction_time = datetime.datetime.fromisoformat(row['last_interaction_time'])
                except (ValueError, KeyError) as e:
                    logger.error(f"Ошибка парсинга/доступа к времени для чата {chat_id}: {e}. Используется текущее время.")
                    last_interaction_time = datetime.datetime.utcnow()

                # Получаем новые поля, обрабатываем возможный KeyError, если миграция не прошла
                scenario_id = row['scenario_id'] if 'scenario_id' in row.keys() else None
                message_counter = row['message_counter'] if 'message_counter' in row.keys() else 0
                if scenario_id is None and 'scenario_id' not in row.keys():
                     logger.warning(f"Колонка scenario_id не найдена в БД для чата {chat_id}, возможно, старая БД.")
                if message_counter == 0 and 'message_counter' not in row.keys():
                     logger.warning(f"Колонка message_counter не найдена в БД для чата {chat_id}, возможно, старая БД.")


                state = ChatState(
                    chat_id=row['chat_id'],
                    user_id=row['user_id'],
                    scenario_id=scenario_id, # Добавлено
                    system_prompt=row['system_prompt'],
                    short_term_memory=stm_instance,
                    ltm_collection_id=row['ltm_collection_id'],
                    last_interaction_time=last_interaction_time,
                    message_counter=message_counter # Добавлено
                )
                logger.debug(f"Состояние чата {chat_id} загружено из БД (Scenario: {scenario_id}).")
                return state
            else:
                logger.debug(f"Состояние для чата {chat_id} не найдено в БД.")
                return None
    except sqlite3.Error as e:
        logger.error(f"Ошибка SQLite при загрузке состояния чата {chat_id}: {e}", exc_info=True)
        return None
    except Exception as e:
         logger.error(f"Непредвиденная ошибка при загрузке чата {chat_id}: {e}", exc_info=True)
         return None

def delete_chat_state(chat_id: int):
    """Удаляет состояние чата из базы данных."""
    query = "DELETE FROM chat_states WHERE chat_id = ?"
    try:
        with sqlite3.connect(SQLITE_DB_NAME, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (chat_id,))
            conn.commit()
            # Проверяем, была ли строка удалена
            if cursor.rowcount > 0:
                logger.info(f"Запись чата {chat_id} удалена из БД.")
                return True # Возвращаем успех
            else:
                 logger.warning(f"Попытка удалить несуществующую запись чата {chat_id} из БД.")
                 return False # Запись не найдена
    except sqlite3.Error as e:
        logger.error(f"Ошибка SQLite при удалении состояния чата {chat_id}: {e}", exc_info=True)
        return False # Ошибка

# --- Вызов инициализации при импорте модуля ---
try:
    initialize_database()
except Exception as init_db_e:
     logger.critical(f"Не удалось инициализировать базу данных: {init_db_e}. Работа бота невозможна.", exc_info=True)
     exit() # Завершаем работу, если БД не инициализировалась

# get_all_chat_ids() - пока не используется, можно удалить или оставить