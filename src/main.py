from fastapi import FastAPI, HTTPException
from src.data import IrisData, IrisResponse
from src.predict import predict

app = FastAPI(
    title="Iris Classifier API",
    description="A FastAPI app to classify Iris flowers using a Decision Tree Classifier.",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Iris Classifier API! Go to /docs to test the endpoints."}

@app.post("/predict", response_model=IrisResponse)
async def predict_species(data: IrisData):
    try:
        result = predict(data)
        return IrisResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
