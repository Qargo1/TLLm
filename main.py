# main.py
import asyncio
import signal
import sys
import html
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ApplicationBuilder,
    Defaults,
    CallbackContext,
    ContextTypes,
    CallbackQueryHandler # Добавили CallbackQueryHandler
)
from telegram.constants import ParseMode

# --- Импорт конфигурации и компонентов ---
from config import TELEGRAM_BOT_TOKEN, logger, SCENARIOS # Импортируем сценарии
from llm.llama_loader import load_llm_model, unload_llm_model, model as llm_model # Импортируем модель для проверки
from memory.embeddings import load_embedding_model, unload_embedding_model, embedding_model # Импортируем модель для проверки
from memory.vector_store import VectorStoreManager
from persistence.chat_manager import ChatManager
from persistence.database import save_chat_state # Импортируем, т.к. может понадобиться напрямую
# Импортируем обработчик текстовых сообщений
from telegram_handlers.message_handlers import handle_message

# --- Глобальные переменные и константы ---
telegram_app: Application | None = None
DELETE_CHAT_CONFIRM_CALLBACK = "delete_chat_confirm_"
DELETE_CHAT_CANCEL_CALLBACK = "delete_chat_cancel_"
SELECT_SCENARIO_CALLBACK = "select_scenario_" # Для кнопок выбора сценария

# --- Расширенный Контекст ---
class CustomContext(CallbackContext[dict, dict, dict]):
    """Кастомный контекст для доступа к менеджерам."""
    def __init__(self, application: Application, chat_id: int = None, user_id: int = None):
        super().__init__(application=application, chat_id=chat_id, user_id=user_id)
        self._chat_manager: Optional[ChatManager] = None
        self._vector_store_manager: Optional[VectorStoreManager] = None

    @property
    def chat_manager(self) -> ChatManager:
        if self._chat_manager is None:
            self._chat_manager = self.application.bot_data.get('chat_manager')
            if self._chat_manager is None:
                logger.error("ChatManager не найден в context.application.bot_data!")
                raise RuntimeError("ChatManager не инициализирован.")
        return self._chat_manager

    @property
    def vector_store_manager(self) -> VectorStoreManager:
        # Хотя он не используется напрямую в хендлерах сейчас, оставим для возможного будущего использования
        if self._vector_store_manager is None:
            self._vector_store_manager = self.application.bot_data.get('vector_store_manager')
            if self._vector_store_manager is None:
                logger.error("VectorStoreManager не найден в context.application.bot_data!")
                raise RuntimeError("VectorStoreManager не инициализирован.")
        return self._vector_store_manager

# --- Инициализация и Остановка ---
async def post_init(application: Application) -> None:
    """Инициализация менеджеров и загрузка моделей."""
    # 1. Модель эмбеддингов
    logger.info("Загрузка модели эмбеддингов...")
    try:
        loaded_emb_model = await asyncio.to_thread(load_embedding_model)
        application.bot_data['embedding_model_loaded'] = loaded_emb_model is not None
        if not application.bot_data['embedding_model_loaded']:
            logger.error("Не удалось загрузить модель эмбеддингов! LTM и рефлексия будут недоступны.")
    except Exception as e:
        logger.critical(f"Критическая ошибка при загрузке модели эмбеддингов: {e}", exc_info=True)
        application.bot_data['embedding_model_loaded'] = False

    # 2. VectorStoreManager
    logger.info("Инициализация VectorStoreManager...")
    try:
        vector_store_manager = VectorStoreManager()
        application.bot_data['vector_store_manager'] = vector_store_manager
    except Exception as e:
        logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации VectorStoreManager: {e}", exc_info=True)
        application.bot_data['vector_store_manager'] = None

    # 3. ChatManager
    logger.info("Инициализация ChatManager...")
    vs_manager = application.bot_data.get('vector_store_manager')
    if vs_manager:
        chat_manager = ChatManager(vector_store_manager=vs_manager)
        application.bot_data['chat_manager'] = chat_manager
    else:
        logger.critical("ChatManager не может быть инициализирован без VectorStoreManager. Запуск невозможен.")
        # Лучше остановить приложение здесь или в main()
        # Пока просто установим None, проверка будет в main()
        application.bot_data['chat_manager'] = None

    from config import (
        GENERATION_MAX_NEW_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P,
        GENERATION_TOP_K, GENERATION_REPETITION_PENALTY
    )
    application.bot_data['GENERATION_MAX_NEW_TOKENS'] = GENERATION_MAX_NEW_TOKENS
    application.bot_data['GENERATION_TEMPERATURE'] = GENERATION_TEMPERATURE
    application.bot_data['GENERATION_TOP_P'] = GENERATION_TOP_P
    application.bot_data['GENERATION_TOP_K'] = GENERATION_TOP_K
    application.bot_data['GENERATION_REPETITION_PENALTY'] = GENERATION_REPETITION_PENALTY
    logger.debug("Параметры генерации добавлены в bot_data.")

    # 4. Загрузка LLM
    logger.info("Загрузка LLM модели...")
    try:
        # Загружаем, только если все предыдущие шаги успешны
        if application.bot_data.get('chat_manager') and application.bot_data.get('vector_store_manager'):
            loaded_llm = await asyncio.to_thread(load_llm_model)
            application.bot_data['llm_loaded'] = loaded_llm is not None
            if not application.bot_data['llm_loaded']:
                 logger.error("Не удалось загрузить LLM модель! Бот не сможет отвечать.")
        else:
            logger.error("Пропуск загрузки LLM из-за ошибок инициализации менеджеров.")
            application.bot_data['llm_loaded'] = False
    except Exception as e:
        logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке LLM: {e}", exc_info=True)
        application.bot_data['llm_loaded'] = False


async def pre_shutdown(application: Application) -> None:
    """Остановка бота: сохранение чатов, выгрузка моделей."""
    logger.info("Начало процедуры остановки бота...")
    chat_manager: ChatManager = application.bot_data.get('chat_manager')
    if chat_manager:
        logger.info("Сохранение активных состояний чатов...")
        chat_manager.save_all_active_chats()

    if application.bot_data.get('llm_loaded', False):
        logger.info("Выгрузка LLM модели...")
        unload_llm_model() # Ошибки логируются внутри

    if application.bot_data.get('embedding_model_loaded', False):
        logger.info("Выгрузка модели эмбеддингов...")
        unload_embedding_model() # Ошибки логируются внутри

    logger.info("Процедура остановки бота завершена.")

# --- Обработчики Команд ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветствие и кнопки выбора сценария."""
    user = update.effective_user
    logger.info(f"Пользователь {user.username} (ID: {user.id}) запустил бота.")

    keyboard = [
        [InlineKeyboardButton(f"{idx}. {info['name']}", callback_data=f"{SELECT_SCENARIO_CALLBACK}{idx}")]
        for idx, info in SCENARIOS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    welcome_message = (
        f"Привет, {user.mention_html()}!\n\n"
        "Я - ролевой чат-бот. Выбери один из сценариев для начала:"
    )
    await update.message.reply_html(welcome_message, reply_markup=reply_markup)

async def select_scenario_callback(update: Update, context: CustomContext) -> None:
    """Обрабатывает выбор сценария через кнопку."""
    query = update.callback_query
    await query.answer() # Отвечаем на коллбэк

    try:
        scenario_id = int(query.data.split(SELECT_SCENARIO_CALLBACK)[1])
        if scenario_id not in SCENARIOS:
            await query.edit_message_text("Ошибка: Неверный выбор сценария.")
            return

        chat_id = query.message.chat_id
        user_id = query.from_user.id
        scenario_info = SCENARIOS[scenario_id]
        logger.info(f"Пользователь {user_id} в чате {chat_id} выбрал сценарий {scenario_id}: '{scenario_info['name']}'")

        # Используем ChatManager для установки/обновления сценария
        # Метод set_or_get_initial_state должен быть в ChatManager
        chat_state = await context.chat_manager.set_scenario(
            chat_id=chat_id,
            user_id=user_id,
            scenario_id=scenario_id,
            initial_prompt=scenario_info['initial_system_prompt']
        )

        # Отправляем подтверждение и, возможно, первое сообщение сценария
        await query.edit_message_text(
            f"Выбран сценарий: <b>{html.escape(scenario_info['name'])}</b>.\n"
            f"Начинаем...",
            parse_mode=ParseMode.HTML
        )

        # --- Опционально: Запросить у LLM первое сообщение/описание ---
        # Это можно сделать, если initial_prompt только задает сцену,
        # а первое сообщение бота должно быть сгенерировано.
        # Например, вызвать handle_message с пустым user_prompt?
        # Или добавить специальную логику в ChatManager.set_scenario.
        # Пока просто подтверждаем выбор.
        first_bot_message = "..." # TODO: Сгенерировать первое сообщение?
        # await context.bot.send_message(chat_id, first_bot_message)

    except (IndexError, ValueError):
        logger.error(f"Ошибка парсинга callback_data сценария: {query.data}")
        await query.edit_message_text("Ошибка обработки выбора.")
    except Exception as e:
        logger.error(f"Ошибка при выборе сценария {query.data} для чата {query.message.chat_id}: {e}", exc_info=True)
        await query.edit_message_text("Произошла ошибка при выборе сценария.")


async def delete_scenario_command(update: Update, context: CustomContext) -> None:
    """Запрашивает подтверждение для сброса текущего сценария."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    logger.warning(f"Пользователь {user_id} в чате {chat_id} инициировал сброс сценария (/delete).")

    # Получаем текущее состояние, чтобы узнать, есть ли активный сценарий
    # Не создаем новое состояние, если его нет
    current_state = await context.chat_manager.get_existing_chat_state(chat_id)

    if not current_state or not current_state.scenario_id: # Должно быть поле scenario_id в ChatState
         await update.message.reply_text("Нет активного сценария для сброса. Выберите сценарий с помощью /start.")
         return

    scenario_name = SCENARIOS.get(current_state.scenario_id, {}).get("name", f"ID {current_state.scenario_id}")

    keyboard = [
        [
            InlineKeyboardButton("✅ Да, сбросить", callback_data=f"{DELETE_CHAT_CONFIRM_CALLBACK}{chat_id}"),
            InlineKeyboardButton("❌ Отмена", callback_data=f"{DELETE_CHAT_CANCEL_CALLBACK}{chat_id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"Вы уверены, что хотите сбросить текущий сценарий '<b>{html.escape(scenario_name)}</b>'?\n"
        "⚠️ Это действие удалит всю память (STM и LTM) для этого сценария и вернет его к началу.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )

async def delete_scenario_callback(update: Update, context: CustomContext) -> None:
    """Обрабатывает подтверждение/отмену сброса сценария."""
    query = update.callback_query
    await query.answer()

    callback_data = query.data
    try:
        chat_id = int(callback_data.split("_")[-1])
    except (IndexError, ValueError):
        logger.error(f"Некорректный chat_id в delete callback: {callback_data}")
        await query.edit_message_text("Произошла ошибка.")
        return

    if callback_data.startswith(DELETE_CHAT_CONFIRM_CALLBACK):
        logger.info(f"Пользователь подтвердил сброс сценария для чата {chat_id}.")
        try:
            # Метод для сброса/удаления должен быть в ChatManager
            deleted = await context.chat_manager.delete_chat_session(chat_id)
            if deleted:
                await query.edit_message_text(
                    text="Сценарий сброшен. Вся память удалена.\n"
                         "Используйте /start, чтобы выбрать новый сценарий."
                )
            else:
                 await query.edit_message_text(
                    text="Сценарий не найден или уже был сброшен."
                )
        except Exception as e:
            logger.error(f"Ошибка при сбросе сценария чата {chat_id}: {e}", exc_info=True)
            await query.edit_message_text("Не удалось сбросить сценарий.")

    elif callback_data.startswith(DELETE_CHAT_CANCEL_CALLBACK):
        logger.info(f"Пользователь отменил сброс сценария для чата {chat_id}.")
        await query.edit_message_text("Сброс сценария отменен.")


# --- Обработчик Сигналов ---
async def handle_exit_signal(sig, frame):
    """Обработчик сигналов ОС (SIGINT, SIGTERM) для корректной остановки."""
    global telegram_app
    if telegram_app and telegram_app.is_running:
        logger.warning(f"Получен сигнал {sig}. Инициирую штатную остановку бота...")
        # Создаем задачу для асинхронной остановки
        asyncio.create_task(telegram_app.shutdown())
    elif telegram_app:
         logger.warning(f"Получен сигнал {sig}, но приложение не запущено. Пробую остановить.")
         # Попытка остановить, даже если is_running=False (на всякий случай)
         asyncio.create_task(telegram_app.stop())
         # Даем немного времени на завершение и выходим
         await asyncio.sleep(1)
         sys.exit(0)
    else:
        logger.warning(f"Получен сигнал {sig}, но приложение бота не инициализировано. Выход.")
        sys.exit(0)

# --- Основная Функция ---
def main() -> None:
    """Запуск бота."""
    global telegram_app
    if not TELEGRAM_BOT_TOKEN:
        # Логирование уже есть в config.py
        sys.exit(1)

    logger.info("Инициализация Telegram бота...")
    defaults = Defaults(parse_mode=ParseMode.HTML)
    context_types = ContextTypes(context=CustomContext)

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .defaults(defaults)
        .context_types(context_types)
        .post_init(post_init)
        .pre_shutdown(pre_shutdown)
        .build()
    )
    telegram_app = application

    # --- Проверка инициализации ---
    # Убедимся, что критически важные компоненты загрузились
    if not application.bot_data.get('chat_manager') or \
       not application.bot_data.get('vector_store_manager') or \
       not application.bot_data.get('llm_loaded'):
       logger.critical("Критическая ошибка инициализации менеджеров или LLM. Запуск отменен.")
       # Можно добавить более детальную информацию о том, что именно не загрузилось
       sys.exit(1)
    if not application.bot_data.get('embedding_model_loaded'):
        logger.warning("Модель эмбеддингов не загружена - LTM и рефлексия будут недоступны.")


    # --- Регистрация Обработчиков ---
    # Команды
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("delete", delete_scenario_command))
    # Команда /scenario будет обрабатываться через CallbackQueryHandler выбора сценария

    # Обработчики кнопок (CallbackQuery)
    application.add_handler(CallbackQueryHandler(select_scenario_callback, pattern=f"^{SELECT_SCENARIO_CALLBACK}"))
    application.add_handler(CallbackQueryHandler(delete_scenario_callback, pattern=f"^{DELETE_CHAT_CONFIRM_CALLBACK}|^{DELETE_CHAT_CANCEL_CALLBACK}"))

    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Все обработчики зарегистрированы.")

    # --- Обработка Сигналов ---
    if sys.platform != "win32": # SIGINT/SIGTERM могут не работать ожидаемо на Windows в некоторых случаях
        signal.signal(signal.SIGINT, handle_exit_signal)
        signal.signal(signal.SIGTERM, handle_exit_signal)
    else:
         logger.warning("Запуск на Windows. Корректная остановка через Ctrl+C может потребовать нескольких нажатий.")

    # Запуск бота
    logger.info("Запуск бота...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

    # Этот код выполнится после остановки run_polling
    logger.info("Бот остановлен.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Необработанное исключение в main: {e}", exc_info=True)
        sys.exit(1)