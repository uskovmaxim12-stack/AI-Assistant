# server/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import os

app = FastAPI()

# Разрешаем запросы с любых источников (для клиента)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Выбираем модель (можно изменить на более мощную)
MODEL_NAME = "microsoft/DialoGPT-medium"  # или "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Загрузка модели {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Устанавливаем pad_token = eos_token для корректной генерации
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Модель загружена на {device}")

class ChatRequest(BaseModel):
    message: str
    history: list = []  # история сообщений для контекста

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Формируем промпт с историей (для DialoGPT просто добавляем сообщения)
    # Для других моделей нужен свой формат
    if "dialogpt" in MODEL_NAME.lower():
        # DialoGPT ожидает диалог, разделённый токенами EOS
        prompt = tokenizer.eos_token.join(req.history + [req.message]) + tokenizer.eos_token
    else:
        # Для TinyLlama используем шаблон
        prompt = f"<|user|>\n{req.message}\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только новое сообщение (для DialoGPT ответ после последнего EOS)
    if "dialogpt" in MODEL_NAME.lower():
        # Ищем последнее вхождение ответа
        parts = reply.split(tokenizer.eos_token)
        if len(parts) > len(req.history) + 1:
            reply = parts[-1].strip()
        else:
            reply = parts[-1].strip() if parts else ""
    else:
        # Для TinyLlama берём после <|assistant|>
        if "<|assistant|>" in reply:
            reply = reply.split("<|assistant|>")[-1].strip()
    
    if not reply:
        reply = "Извините, не могу ответить."
    
    return ChatResponse(reply=reply)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
