from fastapi import FastAPI
from api.predict import router
from service.predict_service import prediction_service
import uvicorn

app = FastAPI(
    title="Tomato Plant Disease Prediction API",
    description="AI-powered API for detecting diseases in tomato plants",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    """Load the ML model on startup"""
    prediction_service.load_model()

app.include_router(router, tags=["Prediction"])

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)