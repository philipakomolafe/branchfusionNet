from typing import Any, Dict
import logging 
import pathlib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



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
        """Complete prediction pipeline from image to result."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        processed_input = self.preprocess(image)
        raw_output = self.predict(processed_input)
        return self.postprocess(raw_output)
    
# Instantiate the Predictor
prediction_service = Predictor(MODEL_PATH)