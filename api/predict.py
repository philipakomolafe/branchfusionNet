from fastapi import APIRouter, status, UploadFile, File, HTTPException
from PIL import Image
import io
from service.predict_service import prediction_service

router = APIRouter(prefix="/predict_disease")

@router.get("/")
def home():
    return {"message": "Welcome to the Tomato Plant Disease API service"}

@router.get("/health", status_code=status.HTTP_200_OK)
def check_health():
    return {"status": "ok"}

@router.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    """Predict tomato plant disease from uploaded image."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load model if not already loaded
        if prediction_service.model is None:
            prediction_service.load_model()
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get prediction
        result = prediction_service.predict_from_image(image)
        
        return {
            "success": True,
            "filename": file.filename,
            "prediction": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")










