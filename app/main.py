from fastapi import FastAPI
from app.routers import predict

app = FastAPI()

app.include_router(predict.router)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the MedBERT Prediction API"}
