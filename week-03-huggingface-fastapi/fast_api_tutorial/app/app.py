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

# @app.post("/translate")
# # def translate(text_to_translate: TextToTranslate):
# #     return (pipeline(text_to_translate))
# def translate(text_to_translate: TextToTranslate):
#     return (pipeline('Enter Words to Translate'))
# #     return {"message": pipeline(text_to_translate)}
@app.post("/translate")
def translate(text_to_translate: TextToTranslate):
    return (pipeline('Enter Words to Translate'))
    # return {"message": text_to_translate.input_text}

#Anna'snotes on batch prediction
# class BatchPrediction(BaseModel):
#     input_texts: List[str]

# app = FastAPI()


# @app.post("/predict_batch")
# def predict_batch(batch_prediction: BatchPrediction):
#     predictions = [pipeline(text) for text in batch_prediction.input_texts]
#     return predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)