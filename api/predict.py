from fastapi import APIRouter
from starlette import status


router = APIRouter()

@router.get("/")
def home():
    return "Welcome to the Tomato Plant Disease API service"

@router.post("/health")
def check_health():
    return {"status": status.HTTP_302_FOUND}






