# config.py
class Config:
    vocab_size = 5000       # размер словаря (можно увеличить)
    max_len = 128           # максимальная длина последовательности
    d_model = 256           # размер эмбеддингов
    num_heads = 8           # количество голов внимания
    num_layers = 6          # количество слоёв трансформера
    d_ff = 512              # размер скрытого слоя в Feed-Forward
    dropout = 0.1
    batch_size = 32
    learning_rate = 0.0001
    epochs = 10
