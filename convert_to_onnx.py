# convert_to_onnx.py
# Этот скрипт конвертирует обученную модель в ONNX для использования в браузере
# Требует установленных torch и transformers (но это уже внешние библиотеки, но для конвертации)
import torch
import pickle
from model import GPT
from config import Config

# Загружаем веса
with open('weights.pkl', 'rb') as f:
    weights = pickle.load(f)

# Создаём модель PyTorch (аналог нашей GPT) и переносим веса
# ... (сложная часть, опущена)

# Экспорт в ONNX
dummy_input = torch.randint(0, config.vocab_size, (1, config.max_len))
torch.onnx.export(model, dummy_input, "model.onnx")
