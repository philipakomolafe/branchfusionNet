from typing import Any, Dict
import logging 
import pathlib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
from collections import Counter

logger = logging.getLogger(__name__)
MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" / "MultiBranchFusionNet-0.keras"
class_names = [
    "Grey_leaf_spot_(fungi)",
    "Tomato___Bacterial_spot", 
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot", 
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

class Predictor:
    """Predictor class for tomato plant disease classification."""
    def __init__(self, model_path: str | pathlib.Path):
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        """Load the trained model from the specified path."""
        try:
            self.model = load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess the input image for prediction."""
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    
    def predict(self, processed_input) -> np.ndarray:
        """Run prediction on preprocessed input."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        prediction = self.model.predict(processed_input)
        return prediction

    def postprocess(self, raw_output: np.ndarray) -> Dict[str, Any]:
        """Convert the raw model output into API-friendly response (labels, probabilities)"""
        # Get predicted class probabilities
        probabilities = raw_output[0]  # Remove batch dimension
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Format all class probabilities
        class_probabilities = {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities
        }

    def predict_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """Complete prediction pipeline with pre-validation."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # PRE-VALIDATION STEP
        validation_result = self.validate_tomato_image(image)
        
        if not validation_result["is_valid"]:
            return {
                "success": False,
                "predicted_class": "invalid_input",
                "confidence": 0.0,
                "message": validation_result["reason"],
                "validation_details": validation_result,
                "suggestion": "Please upload a clear image of tomato plant leaves"
            }
        
        # Continue with normal prediction if validation passes
        processed_input = self.preprocess(image)
        raw_output = self.predict(processed_input)
        result = self.postprocess(raw_output)
        
        # Add validation info to successful predictions
        result["validation_passed"] = True
        result["validation_details"] = validation_result
        
        return result
    
    def validate_tomato_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Pre-validation to check if image likely contains plant material
        Returns: {"is_valid": bool, "reason": str, "confidence": float}
        """
        
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        # 1. GREEN DOMINANCE CHECK (plants should have significant green)
        if len(img_array.shape) == 3:  # Color image
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Define green range in HSV
            lower_green = np.array([35, 40, 40])   # Lower bound for green
            upper_green = np.array([85, 255, 255]) # Upper bound for green
            
            # Create mask for green pixels
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green_percentage = np.sum(green_mask > 0) / green_mask.size
            
            if green_percentage < 0.15:  # Less than 15% green pixels
                return {
                    "is_valid": False,
                    "reason": f"Insufficient green content ({green_percentage:.1%}). Expected plant material.",
                    "green_percentage": green_percentage
                }
        
        # 2. TEXTURE COMPLEXITY CHECK (leaves have complex textures)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Calculate image gradient (edge intensity)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_score = np.mean(gradient_magnitude)
        
        if texture_score < 10:  # Very smooth/uniform image
            return {
                "is_valid": False, 
                "reason": f"Image too uniform (texture score: {texture_score:.1f}). Expected detailed leaf structure.",
                "texture_score": texture_score
            }
        
        # 3. SIZE AND ASPECT RATIO CHECK
        height, width = img_array.shape[:2]
        if min(height, width) < 100:
            return {
                "is_valid": False,
                "reason": f"Image too small ({width}x{height}). Minimum 100x100 pixels required.",
                "dimensions": (width, height)
            }
        
        # If all checks pass
        return {
            "is_valid": True,
            "reason": "Image passes pre-validation checks",
            "green_percentage": green_percentage if 'green_percentage' in locals() else 0,
            "texture_score": texture_score
        }

# Instantiate the Predictor
prediction_service = Predictor(MODEL_PATH)