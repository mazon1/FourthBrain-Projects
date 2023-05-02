from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the pre-trained model
model = joblib.load('iris.joblib')

# Define the input data schema
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the endpoint for making predictions
@app.post("/predict")
def predict_species(data: IrisData):
    # Extract the input data as a NumPy array
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    # Make a prediction using the pre-trained model
    prediction = model.predict(input_data)
    # Convert the prediction to a string and return it
    species = str(prediction[0])
    return {"species": species}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)