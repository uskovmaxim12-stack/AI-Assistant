# train.py
import numpy as np
from config import Config
from model import GPT
from tokenizer import Tokenizer
import pickle

def train():
    config = Config()
    # Загружаем текстовые данные (например, файл с диалогами)
    with open('data.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    tokenizer = Tokenizer(config.vocab_size)
    tokenizer.train(texts)
    
    # Сохраняем токенизатор
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    model = GPT(config)
    
    # Подготовка данных (очень упрощённо)
    X, y = [], []
    for text in texts:
        ids = tokenizer.encode(text.strip(), config.max_len)
        X.append(ids[:-1])
        y.append(ids[1:])
    X = np.array(X)
    y = np.array(y)
    
    # Обучение (один шаг для демонстрации)
    logits = model.forward(X)
    loss = cross_entropy(logits, y)  # функция не определена
    # backward и обновление весов опущены
    
    # Сохраняем веса
    weights = {}
    # здесь нужно собрать все параметры
    with open('weights.pkl', 'wb') as f:
        pickle.dump(weights, f)

if __name__ == '__main__':
    train()
