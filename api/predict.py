from typing import Optional
from fastapi import APIRouter, status, UploadFile, File, HTTPException, Query 
from PIL import Image
import io
import logging
from service.predict_service import prediction_service
from service.assistant_service import agri_assistant


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict_disease")

@router.get("/")
def home():
    return {"message": "Welcome to the Tomato Plant Disease API service"}

@router.get("/health", status_code=status.HTTP_200_OK)
def check_health():
    return {"status": "ok"}

@router.post("/predict")
async def predict_plant_disease(
    file: UploadFile = File(...), 
    include_advice: bool = Query(False), region: Optional[str] = Query(None, description="Farmer's region"), 
    language: str = Query("en", description="Response language")):
    """Predict tomato plant disease from uploaded image."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load model if not already loaded
        if prediction_service.model_type is None:
            logger.info("Loading model...")
            prediction_service.load_model()
            logger.info(f"Model loaded successfully: {prediction_service.model_type}")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get prediction - this now returns the clean response directly
        result = prediction_service.predict_from_image(image)

        # If advice is requested, get it from the AI assistant. 
        if include_advice and result.get("success"):
            disease = result.get("disease", "Unknown")
            confidence = result.get("confidence", 0.0)
            result["ai_advice"] = agri_assistant.get_advice(disease, confidence, region, language)


        # Return the clean result directly.
        return result
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="Model file not found. Please ensure the model is available.")
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



@router.get("/model/info")
def get_model_info():
    """Get information about the currently loaded model."""
    try:
        if prediction_service.model_type is None:
            prediction_service.load_model()
        
        model_info = {
            "model_type": prediction_service.model_type,
            "available_classes": 10,  # Fixed number since we know it's 10 classes
            "status": "loaded",
            "ai_assistant": "active" if agri_assistant.is_active() else "inactive"
        }
        

        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")