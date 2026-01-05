import pandas as pd
import re
import json

# Загружаем JSON-файл
with open("result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_messages = []

unvaluable = {"saved_messages",
              "Kirimushroom_pantry",
              "✨ITMO Yapping Club After-Party ✨",
              "Telegram",
              }
# Перебираем все чаты
for chat in data["chats"]["list"]:
    chat_name = chat.get("name", chat.get("type", "Unknown"))
    print(f"Обрабатывается чат: {chat_name}")   # 👈 выводим название чата

    if chat_name not in unvaluable:
        for msg in chat.get("messages", []):
            if msg["type"] == "message":
                all_messages.append({
                    "date": msg.get("date"),
                    "text": msg.get("text"),
                    "caption": msg.get("caption")
                })

# Превращаем в DataFrame
messages = pd.DataFrame(all_messages)

# Функция для извлечения текста
def get_text(row):
    parts = []
    if isinstance(row["text"], list):
        for item in row["text"]:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
    elif pd.notnull(row["text"]):
        parts.append(str(row["text"]))
    if pd.notnull(row.get("caption", None)):
        parts.append(str(row["caption"]))
    return "\n".join(parts).strip()

# Применяем функцию
messages["text"] = messages.apply(get_text, axis=1)

# Оставляем только строки, где текст не пустой
messages = messages[messages["text"] != ""]

# Удаляем эмодзи
messages["text"] = messages["text"].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))

# Переводим в нижний регистр
messages["text"] = messages["text"].str.lower()

# Убираем строки, которые выглядят как имена файлов
messages = messages[~messages["text"].str.match(r'.+\.(pdf|jpg|jpeg|png|gif|docx?|xlsx?|pptx?|txt|zip|rar)$', case=False)]

# Убираем ссылки
messages = messages[~messages["text"].str.contains(r"http|www", case=False, na=False)]

# Берём только нужные колонки
result = messages[["date", "text"]]

# --- Функция для сохранения части DataFrame в Excel ---
def save_partial_excel(df, file_name="result_partial.xlsx", start=None, end=None, n_random=None):
    """
    Сохраняет часть DataFrame в Excel.

    Параметры:
    - df: DataFrame
    - file_name: имя выходного файла
    - start: начальный индекс (включительно)
    - end: конечный индекс (не включительно)
    - n_random: если указано, сохранит n случайных строк
    """
    if n_random is not None:
        df_to_save = df.sample(n=n_random, random_state=42)
    elif start is not None or end is not None:
        df_to_save = df.iloc[start:end]
    else:
        df_to_save = df  # сохраняем весь DataFrame, если ничего не указано

    df_to_save.to_excel(file_name, index=False, engine='openpyxl')
    print(f"Сохранено {len(df_to_save)} строк в файл {file_name}")



save_partial_excel(result, file_name="result.xlsx", start=0, end=5000)

# Сохраняем 50 случайных строк
#save_partial_excel(result, file_name="result_random50.xlsx", n_random=50)

# Сохраняем весь DataFrame (как раньше)
#save_partial_excel(result, file_name="result_full.xlsx")

print("result.xlsx создан")
