# telegram_handlers/command_handlers.py
import html
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, TypeAlias # Добавляем TypeAlias для кастомного контекста
from telegram.constants import ParseMode

# Импортируем зависимости
from config import logger, SCENARIOS # Нужны сценарии для /start
# Импортируем ChatManager для /delete и ChatState для проверки
from persistence.chat_manager import ChatManager
from persistence.chat_state import ChatState # Для проверки в /delete
# Импортируем константы для callback_data из main.py (или определяем их здесь)
from main import ( # Предполагаем, что константы определены в main.py
    SELECT_SCENARIO_CALLBACK,
    DELETE_CHAT_CONFIRM_CALLBACK,
    DELETE_CHAT_CANCEL_CALLBACK
)
# Определяем тип для кастомного контекста (если используется)
# from main import CustomContext # Если CustomContext определен в main
# CtxType: TypeAlias = CustomContext
CtxType: TypeAlias = ContextTypes.DEFAULT_TYPE # Используем стандартный пока

# --- Команда /start ---
async def start_command(update: Update, context: CtxType) -> None:
    """Отправляет приветствие и кнопки выбора сценария."""
    user = update.effective_user
    chat_id = update.effective_chat.id # Получаем chat_id
    logger.info(f"Пользователь {user.username} (ID: {user.id}) запустил /start в чате {chat_id}")

    # Создаем кнопки для каждого сценария из config.py
    keyboard = [
        [InlineKeyboardButton(f"{idx}. {info['name']}", callback_data=f"{SELECT_SCENARIO_CALLBACK}{idx}")]
        for idx, info in SCENARIOS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    welcome_message = (
        f"Привет, {user.mention_html()}!\n\n"
        "Я - ролевой чат-бот. Выбери один из сценариев для начала:"
    )
    # Используем reply_html для поддержки user.mention_html()
    await update.message.reply_html(welcome_message, reply_markup=reply_markup)

# --- Команда /delete ---
async def delete_scenario_command(update: Update, context: CtxType) -> None:
    """Запрашивает подтверждение для сброса текущего сценария."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id # Получаем user_id
    logger.warning(f"Пользователь {user_id} в чате {chat_id} инициировал сброс сценария (/delete).")

    # Получаем ChatManager из bot_data
    chat_manager: ChatManager = context.application.bot_data.get('chat_manager')
    if not chat_manager:
        logger.error(f"ChatManager не найден при вызове /delete для чата {chat_id}")
        await update.message.reply_text("Ошибка: Сервис управления чатами недоступен.")
        return

    # Получаем текущее состояние, чтобы узнать, есть ли активный сценарий
    # Используем get_existing_chat_state, который не создает новый стейт
    # Метод должен быть асинхронным
    current_state: Optional[ChatState] = await chat_manager.get_existing_chat_state(chat_id)

    # Проверяем, есть ли состояние и выбран ли сценарий
    if not current_state or current_state.scenario_id is None:
         await update.message.reply_text("Нет активного сценария для сброса. Выберите сценарий с помощью /start.")
         return

    # Получаем имя сценария из config
    scenario_name = SCENARIOS.get(current_state.scenario_id, {}).get("name", f"ID {current_state.scenario_id}")

    # Создаем кнопки подтверждения
    keyboard = [
        [
            InlineKeyboardButton("✅ Да, сбросить", callback_data=f"{DELETE_CHAT_CONFIRM_CALLBACK}{chat_id}"),
            InlineKeyboardButton("❌ Отмена", callback_data=f"{DELETE_CHAT_CANCEL_CALLBACK}{chat_id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Отправляем сообщение с подтверждением
    await update.message.reply_html( # Используем HTML для форматирования имени
        f"Вы уверены, что хотите сбросить текущий сценарий '<b>{html.escape(scenario_name)}</b>'?\n"
        "⚠️ Это действие удалит всю память (STM и LTM) для этого сценария и вернет его к началу.",
        reply_markup=reply_markup
    )

# --- Обработчики Callback Query (кнопок) ---

async def select_scenario_callback(update: Update, context: CtxType) -> None:
    """Обрабатывает выбор сценария через кнопку."""
    query = update.callback_query
    await query.answer() # Отвечаем на коллбэк, чтобы кнопка перестала "грузиться"

    try:
        # Извлекаем ID сценария из callback_data
        scenario_id = int(query.data.split(SELECT_SCENARIO_CALLBACK)[1])
        if scenario_id not in SCENARIOS:
            logger.warning(f"Получен неверный ID сценария: {scenario_id} от пользователя {query.from_user.id}")
            await query.edit_message_text("Ошибка: Неверный выбор сценария.")
            return

        chat_id = query.message.chat_id
        user_id = query.from_user.id
        scenario_info = SCENARIOS[scenario_id]
        logger.info(f"Пользователь {user_id} в чате {chat_id} выбрал сценарий {scenario_id}: '{scenario_info['name']}'")

        # Получаем ChatManager
        chat_manager: ChatManager = context.application.bot_data.get('chat_manager')
        if not chat_manager:
             logger.error(f"ChatManager не найден при выборе сценария для чата {chat_id}")
             await query.edit_message_text("Ошибка: Сервис управления чатами недоступен.")
             return

        # Используем ChatManager для установки/обновления сценария
        # Метод set_scenario должен быть асинхронным
        chat_state = await chat_manager.set_scenario(
            chat_id=chat_id,
            user_id=user_id,
            scenario_id=scenario_id,
            initial_prompt=scenario_info['initial_system_prompt']
        )

        # Редактируем исходное сообщение с кнопками
        await query.edit_message_text(
            f"Выбран сценарий: <b>{html.escape(scenario_info['name'])}</b>.\n"
            f"Начинаем... Можете написать первое сообщение.",
            parse_mode=ParseMode.HTML
        )

        # --- TODO: Опционально: Запросить у LLM первое сообщение/описание ---
        # if chat_state and not chat_state.get_stm_messages(): # Если история пуста
        #    logger.info(f"Генерация первого сообщения для сценария {scenario_id} в чате {chat_id}")
        #    # Нужно вызвать логику генерации ответа, возможно, через handle_message
        #    # или отдельную функцию. Это усложнение, пока оставим без него.
        #    pass

    except (IndexError, ValueError):
        logger.error(f"Ошибка парсинга callback_data сценария: {query.data}")
        await query.edit_message_text("Ошибка обработки выбора.")
    except Exception as e:
        logger.error(f"Ошибка при выборе сценария {query.data} для чата {query.message.chat_id}: {e}", exc_info=True)
        # Редактируем сообщение, чтобы пользователь знал об ошибке
        try:
            await query.edit_message_text("Произошла ошибка при выборе сценария. Попробуйте еще раз.")
        except Exception as edit_err:
            logger.error(f"Не удалось даже отредактировать сообщение об ошибке: {edit_err}")


async def delete_scenario_callback(update: Update, context: CtxType) -> None:
    """Обрабатывает подтверждение/отмену сброса сценария."""
    query = update.callback_query
    await query.answer()

    callback_data = query.data
    try:
        # Извлекаем chat_id из callback_data
        chat_id = int(callback_data.split("_")[-1])
    except (IndexError, ValueError):
        logger.error(f"Некорректный chat_id в delete callback: {callback_data}")
        await query.edit_message_text("Произошла ошибка.")
        return

    # Получаем ChatManager
    chat_manager: ChatManager = context.application.bot_data.get('chat_manager')
    if not chat_manager:
        logger.error(f"ChatManager не найден при обработке delete callback для чата {chat_id}")
        await query.edit_message_text("Внутренняя ошибка: Сервис управления чатами недоступен.")
        return

    if callback_data.startswith(DELETE_CHAT_CONFIRM_CALLBACK):
        logger.info(f"Пользователь {query.from_user.id} подтвердил сброс сценария для чата {chat_id}.")
        try:
            # Вызываем асинхронный метод сброса
            deleted = await chat_manager.delete_chat_session(chat_id)
            if deleted:
                await query.edit_message_text(
                    text="Сценарий сброшен. Вся память удалена.\n"
                         "Используйте /start, чтобы выбрать новый сценарий."
                )
            else:
                 # Это может произойти, если пользователь нажал кнопку дважды
                 await query.edit_message_text(
                    text="Сценарий не найден или уже был сброшен."
                )
        except Exception as e:
            logger.error(f"Ошибка при сбросе сценария чата {chat_id}: {e}", exc_info=True)
            await query.edit_message_text("Не удалось сбросить сценарий. Попробуйте позже.")

    elif callback_data.startswith(DELETE_CHAT_CANCEL_CALLBACK):
        logger.info(f"Пользователь {query.from_user.id} отменил сброс сценария для чата {chat_id}.")
        await query.edit_message_text("Сброс сценария отменен.")