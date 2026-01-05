from custom_tokens import custom_tokenizer, get_infrequent_tokens, mask_tokens
from indexing import encode
from model import Transformer

import torch
import os

# Загружаем наш текст
text = [str(i).strip('\n') for i in open("data_processing/book_clean.txt")]

# Соединяем в одну строку с разделителем <END>
corpus = ' <END> '.join(text)

# Список спецтокенов
# <END> - разделитель
# <UNK> - редкие слова
spec_tokens = ["<END>", "<UNK>"]

# Токенизация

tokens = custom_tokenizer(corpus, spec_tokens) # разобьём (токенизируем) текст
infreq_tokens = get_infrequent_tokens(tokens, min_count=2) # находим токены, которые встречаются меньше, чем 2 раза
tokens = mask_tokens(tokens, infreq_tokens) # заменим их на <UNK>

print("Всего токенов:", len(tokens))
print("Всего уникальных токенов:", len(set(tokens)))

# Словарь

# Создаём список уникальных токенов из корпуса
# set(tokens) удаляет дубликаты, list() превращает обратно в список
vocab_list = list(set(tokens))

# Создаём словарь "слово -> индекс"
# Каждому уникальному токену присваиваем числовой индекс
word2idx = {w: i for i, w in enumerate(vocab_list)}

# Создаём обратный словарь "индекс -> слово"
# Позволяет по числовому индексу получить исходный токен
idx2word = {i: w for w, i in word2idx.items()}

# кодируем список токенов в тензор целых чисел
enc = encode(tokens, vocab_list)
print(enc)

# Разделение на батчи

# Длина последовательности (количество токенов в одном примере)
seq_len = 16

# Разбиваем закодированный текст на куски по seq_len токенов
# enc - это список или тензор с закодированными токенами
# Создаем список из последовательностей длиной 16 токенов
chunks = [enc[i:i+seq_len] for i in range(0, len(enc), seq_len)]

# Оставляем только полные последовательности (длиной ровно 16)
# Это важно, чтобы все примеры в батче имели одинаковую длину
chunks = [c for c in chunks if len(c) == seq_len]

# Преобразуем список последовательностей в тензор PyTorch
# Получаем тензор размерности (batch_size, seq_len)
# где batch_size - количество 16-токенных последовательностей
src = torch.stack(chunks)  # (batch, seq_len)

# Получаем индекс токена начала последовательности (START OF SEQUENCE)
# Внимание: у вас странное название переменной - SOS обычно означает Start,
# но вы используете <END> токен. Возможно, это опечатка или особенность вашей модели?
SOS_idx = word2idx["<END>"]  # токен старта

# Создаем целевые последовательности (tgt) для обучения
# Для каждой последовательности в src:
# 1. Создаем тензор из одного токена SOS в начале
# 2. Добавляем все токены из src, кроме последнего (src[:, :-1])
# Это стандартный подход для обучения language models:
# модель учится предсказывать следующий токен по предыдущим
tgt = torch.cat([torch.full((src.size(0), 1), SOS_idx), src[:, :-1]], dim=1)


# Маски


# Маска источника (src_mask) 

# Используется в Encoder для игнорирования padding-токенов
# Но в вашем случае, так как у вас нет padding (все последовательности одинаковой длины),
# эта маска просто разрешает всем позициям
src_mask = torch.ones(src.size(0), 1, src.size(1), dtype=torch.bool)

# Размеры: (batch_size, 1, seq_len)
# Например: (1331, 1, 16) - для всех 1331 батчей и всех 16 позиций
# Форма (batch, 1, seq_len) позволяет broadcast при умножении с вниманием
# True/1 означает "разрешить", False/0 - "замаскировать" (игнорировать)

# Если бы у вас был padding, маска выглядела бы так:
# [[[True, True, True, ..., False, False, False]]] - где False для padding токенов


# Маска цели (tgt_mask)

# Используется в Decoder для предотвращения "подглядывания в будущее" (causal masking)
# Каждый токен может видеть только предыдущие токены и себя
tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool)).unsqueeze(0)

# Разберем по частям:
# 1. torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool)
#    Создает квадратную матрицу размером (seq_len, seq_len) из True
#    Например, для seq_len=4:
#    [[True, True, True, True],
#     [True, True, True, True],
#     [True, True, True, True],
#     [True, True, True, True]]

# 2. torch.tril(...) - берет нижний треугольник (lower triangular)
#    Оставляет True только ниже и на главной диагонали
#    После tril:
#    [[True, False, False, False],
#     [True, True,  False, False],
#     [True, True,  True,  False],
#     [True, True,  True,  True]]
#    
#    Это означает:
#    - Токен 0 видит только себя
#    - Токен 1 видит токены 0 и 1
#    - Токен 2 видит токены 0, 1 и 2
#    - Токен 3 видит все токены

# 3. .unsqueeze(0) - добавляет размерность batch
#    Размер становится: (1, seq_len, seq_len)
#    Это позволяет использовать одну маску для всех примеров в батче через broadcast

# Такая маска гарантирует, что при предсказании токена i
# модель использует только токены 0..i-1 (не "подглядывая" в будущее)

# Пример

# Предположим, seq_len = 4:
# tgt: [<SOS>, "привет", "как", "дела"]
# 
# При вычислении внимания для каждого токена:
# - "<SOS>" → видит: [<SOS>]                          (предсказывает "привет")
# - "привет" → видит: [<SOS>, "привет"]               (предсказывает "как")
# - "как" → видит: [<SOS>, "привет", "как"]           (предсказывает "дела")
# - "дела" → видит: [<SOS>, "привет", "как", "дела"] (предсказывает <EOS>)



# Инициализация модели

# Определяем устройство для вычислений (GPU или CPU)
# Приоритет отдается CUDA (GPU Nvidia), если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")  

# Определяем размер словаря
# В языковой модели source и target словари обычно одинаковые
src_vocab = len(vocab_list)  # Размер словаря для encoder
tgt_vocab = len(vocab_list)  # Размер словаря для decoder

# Создаем экземпляр модели Transformer
model = Transformer(src_vocab, tgt_vocab).to(device)

# Функция потерь для многоклассовой классификации
# CrossEntropyLoss объединяет LogSoftmax и NLLLoss
# Используется потому что на выходе модели: (batch_size, seq_len, vocab_size)
# А нам нужно предсказать следующий токен для каждой позиции
criterion = torch.nn.CrossEntropyLoss()

# Оптимизатор Adam - популярный адаптивный метод градиентного спуска
# lr=1e-4 (0.0001) - стандартная начальная скорость обучения для трансформеров
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-4,
    betas=(0.9, 0.98),  # стандартные для трансформеров
    eps=1e-9  # для численной стабильности
)

# Перенос данных на выбранное устройство (GPU/CPU)
# Важно: данные и модель должны быть на одном устройстве
src, tgt = src.to(device), tgt.to(device)

# Перенос масок на устройство
src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)


# Сохранение

# Путь к файлу сохранения
checkpoint_path = "transformer_checkpoint.pth"

# Загрузка, если есть сохранение
start_epoch = 1
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Загружено сохранение: эпоха {start_epoch - 1}")



# Обучение модели

# Количество эпох - сколько раз модель увидит весь набор данных
epochs = 10

# start_epoch позволяет продолжить обучение с сохраненной точки
for epoch in range(start_epoch, epochs + 1):

    model.train()
    
    # Обнуляем градиенты перед каждым шагом обучения
    # Градиенты накапливаются, если их не обнулять
    optimizer.zero_grad()
    
    # Прямой проход: подаем данные в модель
    # Модель возвращает предсказания для каждого токена в последовательности
    # Размер выхода: (batch_size, длина_последовательности, размер_словаря)
    out = model(src, tgt, src_mask, tgt_mask)  # (batch, seq_len, vocab)
    
    # Вычисляем функцию потерь
    # Преобразуем выход модели и целевые токены для CrossEntropyLoss
    # view(-1, tgt_vocab) превращает 3D тензор в 2D: (batch*seq_len, vocab_size)
    # tgt.view(-1) превращает целевые токены в 1D: (batch*seq_len)
    loss = criterion(out.view(-1, tgt_vocab), tgt.view(-1))

    # Обратный проход: вычисляем градиенты
    # PyTorch автоматически вычисляет производные по всей вычислительной графе
    loss.backward()
    
    # Шаг оптимизации: обновляем веса модели
    # Оптимизатор использует вычисленные градиенты для изменения параметров
    optimizer.step()

    # Выводим информацию о текущей эпохе
    # .item() извлекает числовое значение из тензора loss
    print(f"Эпоха {epoch}/{epochs} | Loss: {loss.item():.4f}")

    # Сохранение контрольной точки (checkpoint)
    # Сохраняем состояние обучения, чтобы можно было продолжить позже
    torch.save({
        "epoch": epoch,  # Текущая эпоха
        "model_state_dict": model.state_dict(),  # Веса модели
        "optimizer_state_dict": optimizer.state_dict(),  # Состояние оптимизатора
        "loss": loss.item()  # Значение функции потерь
    }, checkpoint_path)
    print(f"Сохранено в {checkpoint_path}")


# Тест генерации
def generate_from_enc(model, enc_tensor, idx2word, SOS_idx, max_len=10):
    model.eval()
    src = enc_tensor.unsqueeze(0).to(device)  # (1, seq_len)
    src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool, device=device)
    
    generated = torch.full((1, 1), SOS_idx, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = torch.tril(torch.ones(generated.size(1), generated.size(1), dtype=torch.bool, device=device)).unsqueeze(0)
        probs = model(src, generated, src_mask, tgt_mask)

        probs = torch.softmax(probs[:, -1, :] / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
    
    tokens_out = [idx2word[int(i)] for i in generated[0, 1:]]  # без SOS
    return tokens_out


test_seq = chunks[random.randint(0, len(chunks)-1)]
gen_tokens = generate_from_enc(model, test_seq, idx2word, SOS_idx)
print("Сгенерированный текст:", " ".join(gen_tokens))