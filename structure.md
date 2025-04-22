/my_llama_rp_bot
|-- main.py             # Точка входа, инициализация бота, диспетчеры
|-- config.py           # Загрузка конфигурации (токены, пути)
|-- requirements.txt    # Зависимости
|-- .env                # Файл с токенами (в .gitignore)
|-- llm/
|   |-- __init__.py
|   |-- llama_loader.py   # Загрузка и взаимодействие с LLM (transformers/llama-cpp)
|   |-- prompt_builder.py # Сборка промптов
|-- telegram_handlers/
|   |-- __init__.py
|   |-- command_handlers.py # Обработчики команд (/start, /set_scenario, etc.)
|   |-- message_handlers.py # Обработчик текстовых сообщений
|-- memory/
|   |-- __init__.py
|   |-- short_term.py     # STM
|   |-- long_term.py      # LTM
|   |-- vector_store.py   # Абстракция векторной БД (ChromaDB)
|   |-- embeddings.py     # Загрузка и использование embedding-модели
|-- persistence/
|   |-- __init__.py
|   |-- chat_state.py     # Класс ChatState
|   |-- chat_manager.py   # Менеджер активных чатов
|   |-- database.py       # Функции для работы с SQLite
|-- utils/
|   |-- __init__.py
|   |-- helpers.py        # Вспомогательные функции (например, асинхронный вызов LLM)
|-- data/                 # Данные (БД SQLite, файлы ChromaDB) - в .gitignore
|   |-- chat_database.db
|   |-- vector_storage/