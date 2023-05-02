from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
pipeline = pipeline(model="t5-small")
# import joblib
# # model = joblib.load("week-03-huggingface-fastapi/Translator.pkl")
# pipeline = joblib.load("week-03-huggingface-fastapi/Translator.pkl")
# # pipeline = 'fast_api_tutorial/app/model/' + model_name # complete this line with the code to load the pipeline from the local file


app = FastAPI()

class TextToTranslate(BaseModel):
    input_text: str

@app.get("/")
def index():
    return {"message": "Hello World"}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/echo")
def echo(text_to_translate: TextToTranslate):
    return {"message": text_to_translate.input_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)