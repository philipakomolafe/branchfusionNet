from fastapi import APIRouter, status


router = APIRouter()

@router.get("/")
def home():
    return "Welcome to the Tomato Plant Disease API service"

@router.get("/health", status_code=status.HTTP_200_OK)
def check_health():
    return {"status": "ok"}









