from typing import List, Set, Union
from nltk.tokenize import RegexpTokenizer
from collections import Counter

def custom_tokenizer(txt: str, spec_tokens: List[str], pattern: str = r"|\d|\w+|[^\s]") -> List[str]:
    """
    Токенизирует текст с учётом специальных токенов.
    
    Токенизация выполняется с помощью регулярных выражений (NLTK RegexpTokenizer).
    Специальные токены сохраняются как целые единицы и не разбиваются.
    
    Параметры:
    
    txt : str
        Исходный текст для токенизации
    spec_tokens : List[str]
        Список специальных токенов (например, <END>, <UNK>), 
        которые должны оставаться неделимыми
    pattern : str, optional
        Шаблон для токенизации (по умолчанию: слова, числа и знаки пунктуации).
        Для посимвольной токенизации используйте '|.'
    
    Возвращает:
    
    List[str]
        Список токенов
        
    Примечания:
    
    1. Специальные токены добавляются в начало шаблона, поэтому имеют приоритет
    2. Пробелы игнорируются и служат разделителями
    """
    pattern = "|".join(spec_tokens) + pattern

    # создадим объект-токенизатор на основе заданного регулярного выражения
    tokenizer = RegexpTokenizer(pattern)
    # выполним токенизацию
    tokens = tokenizer.tokenize(txt)
    return tokens


def get_infrequent_tokens(tokens: Union[List[str], str], min_count: int) -> List[str]:
    """
    Возвращает токены, которые встречаются реже заданного порога.
    
    Функция подсчитывает частоту встречаемости токенов и возвращает те,
    количество которых меньше или равно указанному порогу.
    
    Параметры:
    
    tokens : Union[List[str], str]
        Если передана строка, подсчёт частоты выполняется посимвольно.
        Если передан список, подсчёт выполняется по токенам.
    min_count : int
        Пороговое значение встречаемости. Токены с частотой <= этому значению
        считаются редкими.
    
    Возвращает:

    List[str]
        Список редких токенов
        Возвращаются уникальные токены (без дубликатов)
    """
    counts = Counter(tokens)  # Подсчитываем частоту каждого токена
    infreq_tokens = set([k for k,v in counts.items() if v <= min_count])  # Фильтруем редкие
    return infreq_tokens  # Возвращаем список редких токенов

def mask_tokens(tokens: List[str], mask: Set[str]) -> List[str]:
    """
    Проходим по всем токенам. 
    Если токен в множестве редких токенов, то он заменяется на <UNK>
    """
    return ["<UNK>" if i in mask else i for i in tokens]

