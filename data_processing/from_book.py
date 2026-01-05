from bs4 import BeautifulSoup
import re

with open('124875312.fb2', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'lxml-xml')
    
    # Находим все теги <section> или <p> с текстом
    paragraphs = []
    for tag in soup.find_all(['p', 'section']):
        if tag.text.strip():
            paragraphs.append(tag.text.strip())
    
    # Объединяем в один текст
    full_text = '\n\n'.join(paragraphs)
    
    # Сохраняем
    with open('book_clean.txt', 'w', encoding='utf-8') as out:
        out.write(full_text)
    
    print(f"Сохранили {len(paragraphs)} абзацев")