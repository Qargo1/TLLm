# config.py
import os
from dotenv import load_dotenv
import logging
import torch # Используем torch для проверки GPU напрямую

# Загружаем переменные окружения из файла .env
load_dotenv()

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO, # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(), # Вывод в консоль
        # logging.FileHandler("bot.log", encoding='utf-8') # Вывод в файл
    ]
)
# Уменьшаем шум от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("hnswlib").setLevel(logging.WARNING) # ChromaDB dependency

logger = logging.getLogger("RolePlayBot") # Даем имя нашему логгеру

# --- Основные настройки ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN не найден в .env файле! Запуск невозможен.")
    exit() # Завершаем работу, если токена нет

# --- Настройки Моделей ---
MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME_OR_PATH", "meta-llama/Meta-Llama-3.1-8B-Instruct")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Настройки Устройства ---
# Проверяем доступность CUDA
try:
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        DEVICE = "cuda"
        # bfloat16 предпочтительнее для Ampere+ GPU, float16 для старых
        TORCH_DTYPE = torch.bfloat16
        logger.info("CUDA доступна. Используется GPU (cuda). Тип данных: bfloat16.")
        try:
            import bitsandbytes
            logger.info("Библиотека bitsandbytes найдена. Квантизация возможна.")
            # Можно добавить флаг для включения квантизации по умолчанию, если нужно
            # USE_QUANTIZATION = True
        except ImportError:
            logger.warning("Библиотека bitsandbytes не найдена. 4/8-битная квантизация через transformers недоступна.")
            # USE_QUANTIZATION = False
    else:
        DEVICE = "cpu"
        TORCH_DTYPE = torch.float32 # CPU обычно работает с float32
        logger.info("CUDA недоступна. Используется CPU. Тип данных: float32.")
        logger.warning("Использование LLM на CPU может быть очень медленным!")
except ImportError:
    logger.critical("PyTorch не найден! Установите PyTorch согласно инструкции. Запуск невозможен.")
    exit()
except Exception as e:
     logger.critical(f"Ошибка при настройке устройства PyTorch: {e}. Запуск невозможен.")
     exit()

# --- Параметры Генерации Текста (Основной Ответ) ---
# Можно тюнить для разных моделей и стилей
try:
    GENERATION_MAX_NEW_TOKENS = int(os.getenv("GENERATION_MAX_NEW_TOKENS", 300)) # Немного увеличим
    GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", 0.75)) # Чуть выше для "живости"
    GENERATION_TOP_P = float(os.getenv("GENERATION_TOP_P", 0.9))
    GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", 50))
    GENERATION_REPETITION_PENALTY = float(os.getenv("GENERATION_REPETITION_PENALTY", 1.1)) # Штраф за повторения
except ValueError as e:
    logger.warning(f"Ошибка чтения параметров генерации из .env: {e}. Используются значения по умолчанию.")
    GENERATION_MAX_NEW_TOKENS = 300
    GENERATION_TEMPERATURE = 0.75
    GENERATION_TOP_P = 0.9
    GENERATION_TOP_K = 50
    GENERATION_REPETITION_PENALTY = 1.1

# --- Настройки Памяти ---
STM_MAX_MESSAGES = int(os.getenv("STM_MAX_MESSAGES", 12)) # Храним 6 пар сообщений
LTM_RETRIEVAL_COUNT = int(os.getenv("LTM_RETRIEVAL_COUNT", 3)) # Количество извлекаемых фактов LTM

# --- Настройки Рефлексии и Динамического Промпта ---
REFLECTION_INTERVAL_MESSAGES = int(os.getenv("REFLECTION_INTERVAL_MESSAGES", 10)) # Каждые 10 сообщений (user+bot)
# Параметры LLM для задачи рефлексии (могут отличаться от основной генерации)
REFLECTION_MAX_NEW_TOKENS = 512 # Больше токенов для анализа и генерации промпта/фактов
REFLECTION_TEMPERATURE = 0.4 # Ниже температура для более фактологичного вывода
REFLECTION_TOP_P = 0.9
REFLECTION_TOP_K = 40
REFLECTION_REPETITION_PENALTY = 1.15 # Немного выше штраф

# Шаблон промпта для рефлексии (будет использоваться в reflection_service.py)
# {dynamic_system_prompt} - текущий динамический промпт
# {recent_history} - последние N сообщений из STM
REFLECTION_PROMPT_TEMPLATE = """
[SYSTEM]
Ты - внутренний модуль анализа диалога для ролевого ИИ. Твоя задача - проанализировать недавний диалог и обновить состояние персонажа и ключевые факты.
Текущее состояние персонажа и сценарий описаны ниже:
<CURRENT_STATE>
{dynamic_system_prompt}
</CURRENT_STATE>

Проанализируй недавний диалог:
<RECENT_DIALOGUE>
{recent_history}
</RECENT_DIALOGUE>

Основываясь на <CURRENT_STATE> и <RECENT_DIALOGUE>, выполни следующие действия:
1.  **Извлеки Ключевые Факты:** Определи 1-3 самых важных новых факта, события, изменения в отношениях или личности персонажа, которые произошли в недавнем диалоге и которые важно запомнить надолго. Выводи каждый факт в формате "FACT: [описание факта]". Если значимых фактов нет, напиши "FACT: None".
2.  **Обнови Состояние Персонажа:** Сгенерируй обновленное описание состояния персонажа и сценария. Оно должно включать изначальную основу из <CURRENT_STATE>, но дополненную новыми нюансами, настроением, целями или деталями, проявившимися в <RECENT_DIALOGUE>. Не меняй кардинально личность, а развивай ее. Обновленное состояние должно быть самодостаточным описанием для LLM, играющей роль. Выведи его внутри тегов <UPDATED_STATE>...</UPDATED_STATE>.

Выводи только запрошенные секции "FACT:" и "<UPDATED_STATE>...</UPDATED_STATE>". Не добавляй никаких других комментариев.
[/SYSTEM]

[USER]
Проанализируй предоставленные данные и сгенерируй факты и обновленное состояние.
[/USER]

[ASSISTANT]
"""

# --- Предустановленные Сценарии ---
# !!! ЗАПОЛНИ ЭТИ СЦЕНАРИИ СВОИМИ ДАННЫМИ !!!
SCENARIOS = {
    1: {
        "name": "Заброшенная Лаборатория", # Краткое имя для кнопки/списка
        "initial_system_prompt": ( # Начальная инструкция для LLM
            "Ты находишься в тускло освещенной заброшенной лаборатории. Воздух пахнет озоном и пылью. "
            "Повсюду разбросаны сломанные колбы и странное оборудование. Неясно, что здесь произошло. "
            "Ты осторожно осматриваешься. Внезапно ты слышишь тихий шорох из-за перевернутого стола..."
            # (Примечание: Бот пока НЕ является персонажем, он начнет импровизировать после первого сообщения пользователя,
            # или можно добавить сюда описание персонажа, например: "Ты - андроид-ассистент AL-7, активировавшийся здесь. Твоя память повреждена.")
        )
    },
    2: {
        "name": "Таверна 'Пьяный Гоблин'",
        "initial_system_prompt": (
            "Ты заходишь в шумную таверну 'Пьяный Гоблин'. Пахнет элем, жареным мясом и табаком. "
            "За столами сидят разношерстные авантюристы, гномы стучат кружками, а бард неумело пытается играть на лютне. "
            "Ты подходишь к стойке, где протирает кружку одноглазый трактирщик с мрачным видом. "
            "Рядом со стойкой, в темном углу, сидит фигура в плаще, внимательно наблюдающая за тобой..."
            # (Примечание: Опять же, бот не определен. LLM может сыграть трактирщика, фигуру в плаще или кого-то еще,
            # или можно указать роль: "Ты - та самая фигура в плаще, эльфийка-изгнанница по имени Лиара. Ты ждешь здесь связного.")
         )
    },
    3: {
        "name": "Киберпанк Расследование",
        "initial_system_prompt": (
            "Неоновые вывески мегакорпораций отражаются в мокром асфальте Ночного Города. Идет дождь. "
            "Ты стоишь в темном переулке, твой старый кибердек потрескивает от помех. Тебе передали зашифрованное сообщение о странном исчезновении. "
            "Нужно найти зацепки. Твой контакт, хакер по прозвищу 'Призрак', должен был ждать тебя здесь, но его нет. Вместо этого из тени появляется девушка с хромированной рукой и подозрительно знакомым взглядом..."
            # (Примечание: Можно задать роль бота: "Ты - та самая девушка, наемница по имени Рейн. Тебя наняли перехватить сообщение, предназначенное 'Призраку'.")
        )
    }
}
# Проверка, что сценариев ровно 3
if len(SCENARIOS) != 3 or not all(k in SCENARIOS for k in [1, 2, 3]):
    logger.critical("Ошибка в SCENARIOS: Должно быть ровно 3 сценария с ключами 1, 2, 3.")
    exit()


# --- Пути к Данным ---
DATA_PATH = os.getenv("DATA_PATH", "data") # Папка для данных
SQLITE_DB_NAME = os.path.join(DATA_PATH, os.getenv("SQLITE_DB_NAME", "roleplay_chats.db"))
VECTOR_STORE_PATH = os.path.join(DATA_PATH, os.getenv("VECTOR_STORE_PATH", "vector_storage"))

# Создаем папки для данных, если их нет
try:
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    logger.info(f"Папки для данных проверены/созданы: {DATA_PATH}")
except OSError as e:
     logger.critical(f"Не удалось создать папки для данных ({DATA_PATH}, {VECTOR_STORE_PATH}): {e}. Запуск невозможен.")
     exit()


logger.info("Конфигурация загружена.")
logger.info(f"LLM: {MODEL_NAME_OR_PATH}, Embeddings: {EMBEDDING_MODEL_NAME}")
logger.info(f"Устройство: {DEVICE}, Тип данных: {TORCH_DTYPE}")
logger.info(f"Доступные сценарии: {[info['name'] for info in SCENARIOS.values()]}")