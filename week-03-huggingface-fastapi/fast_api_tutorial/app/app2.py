from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
translator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

class TextToTranslate(BaseModel):
    input_text: str

@app.get("/")
def index():
    return {"message": "Hello World"}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/translate")
def translate(text_to_translate: TextToTranslate):
    return {"message": translator(text_to_translate.input_text)}

class BatchPrediction(BaseModel):
    input_texts: List[str]

@app.post("/predict_batch")
def predict_batch(batch_prediction: BatchPrediction):
    predictions = [translator(text) for text in batch_prediction.input_texts]
    return predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
