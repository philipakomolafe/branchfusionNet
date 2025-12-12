from typing import Any, Dict
import logging 
import pathlib
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from collections import Counter
import warnings

logger = logging.getLogger(__name__)
# Updated to use TFLite model
TFLITE_MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" / "MultiBranchFusionNet-0.tflite"
KERAS_MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" / "MultiBranchFusionNet-0.keras"
VALIDATOR_MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" / "tomato_not_tomato.keras"

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

# Human-readable disease names
disease_display_names = {
    "Grey_leaf_spot_(fungi)": "Grey Leaf Spot (Fungal)",
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites": "Spider Mites Infestation",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato___healthy": "Healthy Plant"
}

# Suggestions for each disease
disease_treatments = {
    "Grey_leaf_spot_(fungi)": "Remove affected leaves, improve air circulation, and apply fungicide containing chlorothalonil or copper.",
    "Tomato___Bacterial_spot": "Remove infected plant parts, apply copper-based bactericide, and avoid overhead watering.",
    "Tomato___Early_blight": "Remove lower infected leaves, apply fungicide, and mulch around plants to prevent soil splash.",
    "Tomato___Late_blight": "Remove and destroy infected plants IMMEDIATELY. Apply systemic fungicide urgently and isolate affected area.",
    "Tomato___Leaf_Mold": "Increase air circulation, reduce humidity around plants, and apply fungicide spray.",
    "Tomato___Septoria_leaf_spot": "Remove infected lower leaves, apply fungicide, and water at soil level only.",
    "Tomato___Spider_mites": "Spray plants with water to dislodge mites, apply insecticidal soap, and increase humidity.",
    "Tomato___Target_Spot": "Remove infected leaves, apply fungicide, and improve air circulation.",
    "Tomato___Yellow_Leaf_Curl_Virus": "Remove infected plants immediately, control whitefly vectors, and use virus-resistant varieties.",
    "Tomato___healthy": "Continue current care routine. Monitor regularly for early signs of disease and maintain good garden hygiene."
}



# class indices for tomato validator
# {'Not Tomato': 0, 'Tomato': 1}
class TomatoValidator:
    """Validator class to check if image contains a tomato plant."""
    def __init__(self, model_path: pathlib.Path):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.class_names = ["Not Tomato", "Tomato"]
        self.confidence_threshold = 0.7  # Minimum confidence to accept as tomato
    
    def load_model(self) -> None:
        """Load the tomato validator model."""
        if not self.model_path.exists():
            logger.warning(f"Validator model not found at {self.model_path}. Validation will be skipped.")
            return
        
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Tomato validator model loaded from {self.model_path}")
            
            # Log model info
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
            logger.info(f"Validator input shape: {input_shape}, output shape: {output_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load validator model: {e}")
            self.is_loaded = False
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for the validator model."""
        # Adjust target size based on your model's training
        target_size = (224, 224)  # Change if your model uses different size
        
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def validate(self, image: Image.Image) -> Dict[str, Any]:
        """
        Check if image contains a tomato plant.
        
        Returns:
            {
                "is_tomato": bool,
                "confidence": float,
                "message": str
            }
        """
        # If model not loaded, skip validation (allow all images)
        if not self.is_loaded:
            logger.warning("Validator model not loaded. Skipping tomato validation.")
            return {
                "is_tomato": True,
                "confidence": 1.0,
                "message": "Validation skipped (model not available)"
            }
        
        try:
            # Preprocess and predict
            processed = self.preprocess(image)
            prediction = self.model.predict(processed, verbose=0)
            
            # Handle different output formats
            if prediction.shape[-1] == 1:
                # Binary output (sigmoid): value > 0.5 = tomato
                tomato_confidence = float(prediction[0][0])
                is_tomato = tomato_confidence >= self.confidence_threshold
            else:
                # Multi-class output (softmax): [not_tomato, tomato]
                # Adjust index based on your class order
                tomato_idx = self.class_names.index("tomato")
                tomato_confidence = float(prediction[0][tomato_idx])
                is_tomato = tomato_confidence >= self.confidence_threshold
            
            if is_tomato:
                message = f"Image verified as tomato plant with ({tomato_confidence:.1%} confidence score)"
            else:
                message = f"Image does not appear to be a tomato plant with ({tomato_confidence:.1%} confidence score)"
            
            return {
                "is_tomato": is_tomato,
                "confidence": tomato_confidence,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # On error, allow image through with warning
            return {
                "is_tomato": True,
                "confidence": 0.0,
                "message": f"Validation error: {str(e)}. Proceeding with prediction."
            }




class Predictor:
    """Predictor class for tomato plant disease classification using TFLite."""
    
    def __init__(self, tflite_model_path: pathlib.Path, keras_model_path: pathlib.Path = None, validator_model_path: pathlib.Path = None):
        self.tflite_model_path = tflite_model_path
        self.keras_model_path = keras_model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_type = None
        
        # Initialize tomato validator
        self.validator = TomatoValidator(validator_model_path) if validator_model_path else None

    def load_model(self) -> None:
        """Load the TFLite model with fallback to Keras model."""
        try:
            # Load validator model first
            if self.validator:
                self.validator.load_model()
            
            # Try to load TFLite model first
            if self.tflite_model_path.exists():
                self._load_tflite_model()
                return
            
            # Fallback to Keras model if TFLite doesn't exist
            if self.keras_model_path and self.keras_model_path.exists():
                self._load_keras_model()
                return
            
            raise FileNotFoundError(f"Neither TFLite model ({self.tflite_model_path}) nor Keras model found")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            if self.keras_model_path and self.keras_model_path.exists():
                logger.info("Attempting fallback to Keras model...")
                self._load_keras_model()
            else:
                raise
    
    def _load_tflite_model(self) -> None:
        """Load TFLite model using LiteRT or TensorFlow Lite."""
        try:
            try:
                import ai_edge_litert.interpreter as litert_interpreter
                self.interpreter = litert_interpreter.Interpreter(model_path=str(self.tflite_model_path))
                logger.info("Using LiteRT interpreter")
            except ImportError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_path))
                logger.info("Using TensorFlow Lite interpreter")
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_type = "tflite"
            
            logger.info(f"TFLite model loaded successfully from {self.tflite_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise
    
    def _load_keras_model(self) -> None:
        """Load Keras model as fallback."""
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(self.keras_model_path)
            self.model_type = "keras"
            logger.info(f"Keras model loaded successfully from {self.keras_model_path}")
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            raise

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess the input image for prediction."""
        if self.model_type == "tflite":
            input_shape = self.input_details[0]['shape']
            target_size = (input_shape[1], input_shape[2])
        else:
            target_size = (224, 224)
        
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def predict(self, processed_input: np.ndarray) -> np.ndarray:
        """Run prediction on preprocessed input."""
        if self.model_type == "tflite":
            if self.interpreter is None:
                raise RuntimeError("TFLite model not loaded. Call load_model() first.")
            
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_input)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            return prediction
            
        elif self.model_type == "keras":
            if self.model is None:
                raise RuntimeError("Keras model not loaded. Call load_model() first.")
            return self.model.predict(processed_input, verbose=0)
        
        else:
            raise RuntimeError("No model loaded")

    def postprocess(self, raw_output: np.ndarray) -> Dict[str, Any]:
        """Convert the raw model output into API-friendly response."""
        probabilities = raw_output[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        # treatment_info = disease_treatments.get(predicted_class, {})
        
        return { 
            "disease": predicted_class,
            "confidence": confidence,
        }

    def predict_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """Complete prediction pipeline with tomato validation."""
        if self.model_type is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # STEP 1: Validate if image is a tomato plant
        if self.validator and self.validator.is_loaded:
            validation_result = self.validator.validate(image)
            
            if not validation_result["is_tomato"]:
                logger.warning(f"Tomato validation failed: {validation_result['message']}")
                return {
                    "success": False,
                    "message": validation_result["message"],
                    "suggestion": "Please upload a clear image of tomato plant leaves for disease detection"
                }
            
            logger.info(f"Tomato validation passed: {validation_result['message']}")
        
        # STEP 2: Basic image quality validation
        quality_check = self._validate_image_quality(image)
        if not quality_check["is_valid"]:
            return {
                "success": False,
                "message": quality_check["reason"],
                "suggestion": "Ensure the image has good lighting and clear leaf details"
            }
        
        # STEP 3: Run disease prediction
        processed_input = self.preprocess(image)
        raw_output = self.predict(processed_input)
        result = self.postprocess(raw_output)
        
        result["success"] = True
        return result
    
    def _validate_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Basic image quality checks."""
        img_array = np.array(image)
        
        # Size check
        height, width = img_array.shape[:2]
        if min(height, width) < 100:
            return {
                "is_valid": False,
                "reason": f"Image too small ({width}x{height}). Minimum 100x100 pixels required."
            }
        
        # Color check (should have some green for plant material)
        if len(img_array.shape) == 3:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green_percentage = np.sum(green_mask > 0) / green_mask.size
            
            if green_percentage < 0.10:  # Less than 10% green
                return {
                    "is_valid": False,
                    "reason": f"Image lacks sufficient green plant material ({green_percentage:.1%})"
                }
        
        return {"is_valid": True, "reason": "Image quality acceptable"}


# Instantiate the Predictor with TFLite model and Keras fallback
prediction_service = Predictor(TFLITE_MODEL_PATH, KERAS_MODEL_PATH, VALIDATOR_MODEL_PATH)