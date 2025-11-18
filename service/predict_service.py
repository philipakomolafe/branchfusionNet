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

# Treatment recommendations for each disease
disease_treatments = {
    "Grey_leaf_spot_(fungi)": {
        "disease_name": "Grey Leaf Spot (Fungal)",
        "severity": "moderate",
        "treatment": {
            "immediate": [
                "Remove affected leaves immediately",
                "Improve air circulation around plants",
                "Apply fungicide containing chlorothalonil or copper"
            ],
            "preventive": [
                "Water at soil level, avoid wetting leaves",
                "Space plants properly for airflow",
                "Apply mulch to prevent soil splash",
                "Rotate crops annually"
            ],
            "organic": [
                "Use neem oil spray",
                "Apply baking soda solution (1 tsp per quart water)",
                "Use compost tea as foliar spray"
            ]
        },
        "prognosis": "Good if caught early and treated promptly"
    },
    "Tomato___Bacterial_spot": {
        "disease_name": "Bacterial Spot",
        "severity": "high",
        "treatment": {
            "immediate": [
                "Remove and destroy infected plant parts",
                "Apply copper-based bactericide",
                "Avoid overhead watering"
            ],
            "preventive": [
                "Use disease-resistant varieties",
                "Sanitize tools between plants",
                "Improve drainage and air circulation",
                "Avoid working with wet plants"
            ],
            "organic": [
                "Use copper sulfate spray",
                "Apply beneficial bacteria (Bacillus subtilis)",
                "Maintain proper soil pH (6.0-6.8)"
            ]
        },
        "prognosis": "Manageable with proper treatment, but can spread rapidly"
    },
    "Tomato___Early_blight": {
        "disease_name": "Early Blight",
        "severity": "moderate",
        "treatment": {
            "immediate": [
                "Remove lower infected leaves",
                "Apply fungicide (chlorothalonil, mancozeb)",
                "Stake plants to improve air circulation"
            ],
            "preventive": [
                "Mulch around plants to prevent soil splash",
                "Water at base of plants",
                "Prune lower branches",
                "Practice crop rotation"
            ],
            "organic": [
                "Use baking soda spray",
                "Apply neem oil regularly",
                "Use copper fungicide",
                "Compost tea applications"
            ]
        },
        "prognosis": "Very treatable when detected early"
    },
    "Tomato___Late_blight": {
        "disease_name": "Late Blight",
        "severity": "critical",
        "treatment": {
            "immediate": [
                "Remove and destroy infected plants IMMEDIATELY",
                "Apply systemic fungicide urgently",
                "Isolate affected area"
            ],
            "preventive": [
                "Monitor weather for cool, wet conditions",
                "Use resistant varieties",
                "Ensure excellent drainage",
                "Avoid overhead irrigation"
            ],
            "organic": [
                "Copper-based treatments (limited effectiveness)",
                "Remove infected plants immediately",
                "Focus on prevention rather than treatment"
            ]
        },
        "prognosis": "Serious disease - can destroy entire crops quickly. Immediate action required."
    },
    "Tomato___Leaf_Mold": {
        "disease_name": "Leaf Mold",
        "severity": "moderate",
        "treatment": {
            "immediate": [
                "Increase air circulation immediately",
                "Reduce humidity around plants",
                "Apply fungicide spray"
            ],
            "preventive": [
                "Space plants properly",
                "Use drip irrigation",
                "Ensure good ventilation in greenhouses",
                "Remove lower leaves regularly"
            ],
            "organic": [
                "Use baking soda solution",
                "Apply neem oil",
                "Improve air circulation naturally",
                "Use beneficial microorganisms"
            ]
        },
        "prognosis": "Easily managed with proper environmental controls"
    },
    "Tomato___Septoria_leaf_spot": {
        "disease_name": "Septoria Leaf Spot",
        "severity": "moderate",
        "treatment": {
            "immediate": [
                "Remove infected lower leaves",
                "Apply fungicide containing chlorothalonil",
                "Improve plant spacing"
            ],
            "preventive": [
                "Mulch to prevent soil splash",
                "Water at soil level only",
                "Stake and prune plants properly",
                "Practice crop rotation"
            ],
            "organic": [
                "Neem oil applications",
                "Copper fungicide spray",
                "Baking soda solution",
                "Remove debris regularly"
            ]
        },
        "prognosis": "Good recovery with consistent treatment"
    },
    "Tomato___Spider_mites": {
        "disease_name": "Spider Mites Infestation",
        "severity": "moderate",
        "treatment": {
            "immediate": [
                "Spray plants with water to dislodge mites",
                "Apply miticide or insecticidal soap",
                "Increase humidity around plants"
            ],
            "preventive": [
                "Maintain adequate moisture levels",
                "Introduce predatory mites",
                "Avoid excessive nitrogen fertilization",
                "Regular monitoring of leaf undersides"
            ],
            "organic": [
                "Neem oil spray",
                "Insecticidal soap solution",
                "Introduce ladybugs or lacewings",
                "Use reflective mulch"
            ]
        },
        "prognosis": "Excellent with early detection and treatment"
    },
    "Tomato___Target_Spot": {
        "disease_name": "Target Spot",
        "severity": "moderate",
        "treatment": {
            "immediate": [
                "Remove infected leaves",
                "Apply fungicide treatment",
                "Improve air circulation"
            ],
            "preventive": [
                "Use drip irrigation",
                "Mulch around plants",
                "Proper plant spacing",
                "Rotate crops annually"
            ],
            "organic": [
                "Copper-based fungicide",
                "Baking soda spray",
                "Neem oil application",
                "Compost tea foliar spray"
            ]
        },
        "prognosis": "Good response to treatment when caught early"
    },
    "Tomato___Yellow_Leaf_Curl_Virus": {
        "disease_name": "Yellow Leaf Curl Virus",
        "severity": "critical",
        "treatment": {
            "immediate": [
                "Remove infected plants immediately",
                "Control whitefly vectors",
                "Isolate healthy plants"
            ],
            "preventive": [
                "Use virus-resistant varieties",
                "Control whitefly populations",
                "Use reflective mulch",
                "Install fine mesh screens"
            ],
            "organic": [
                "Yellow sticky traps for whiteflies",
                "Neem oil for vector control",
                "Remove weeds that harbor virus",
                "Use companion planting"
            ]
        },
        "prognosis": "No cure available - focus on prevention and vector control"
    },
    "Tomato___healthy": {
        "disease_name": "Healthy Plant",
        "severity": "none",
        "treatment": {
            "immediate": [
                "Continue current care routine",
                "Monitor regularly for early signs of disease"
            ],
            "preventive": [
                "Maintain consistent watering",
                "Ensure proper nutrition",
                "Monitor for pests regularly",
                "Practice good garden hygiene"
            ],
            "organic": [
                "Continue organic practices",
                "Use compost for soil health",
                "Encourage beneficial insects",
                "Regular soil testing"
            ]
        },
        "prognosis": "Plant is healthy - maintain current practices"
    }
}

class Predictor:
    """Predictor class for tomato plant disease classification using TFLite."""
    def __init__(self, tflite_model_path: str | pathlib.Path, keras_model_path: str | pathlib.Path = None):
        self.tflite_model_path = tflite_model_path
        self.keras_model_path = keras_model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_type = None

    def load_model(self) -> None:
        """Load the TFLite model with fallback to Keras model."""
        try:
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
            # Try fallback to Keras model
            if self.keras_model_path and self.keras_model_path.exists():
                logger.info("Attempting fallback to Keras model...")
                self._load_keras_model()
            else:
                raise
    
    def _load_tflite_model(self) -> None:
        """Load TFLite model using LiteRT or TensorFlow Lite."""
        try:
            # Try using LiteRT first (new TensorFlow Lite)
            try:
                import ai_edge_litert.interpreter as litert_interpreter
                self.interpreter = litert_interpreter.Interpreter(model_path=str(self.tflite_model_path))
                logger.info("Using LiteRT interpreter")
            except ImportError:
                # Fallback to TensorFlow Lite with warning suppression
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_path))
                logger.info("Using TensorFlow Lite interpreter")
            
            self.interpreter.allocate_tensors()
            
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_type = "tflite"
            
            logger.info(f"TFLite model loaded successfully from {self.tflite_model_path}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")
            
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
            # Get input shape from TFLite model
            input_shape = self.input_details[0]['shape']
            target_size = (input_shape[1], input_shape[2])  # (height, width)
        else:
            target_size = (224, 224)  # Default for Keras model
        
        # Resize and normalize
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    
    def predict(self, processed_input: np.ndarray) -> np.ndarray:
        """Run prediction on preprocessed input."""
        if self.model_type == "tflite":
            if self.interpreter is None:
                raise RuntimeError("TFLite model not loaded. Call load_model() first.")
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_input)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            return prediction
            
        elif self.model_type == "keras":
            if self.model is None:
                raise RuntimeError("Keras model not loaded. Call load_model() first.")
            return self.model.predict(processed_input, verbose=0)
        
        else:
            raise RuntimeError("No model loaded")

    def postprocess(self, raw_output: np.ndarray) -> Dict[str, Any]:
        """Convert the raw model output into API-friendly response with treatment recommendations."""
        # Get predicted class probabilities
        probabilities = raw_output[0]  # Remove batch dimension
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Get treatment information
        treatment_info = disease_treatments.get(predicted_class, {})
        
        return {
            "disease": predicted_class,
            "confidence": confidence,
            "disease_info": treatment_info,
            "model_type": self.model_type
        }

    def predict_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """Complete prediction pipeline with internal pre-validation."""
        if self.model_type is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # INTERNAL PRE-VALIDATION (not exposed in response)
        validation_result = self.validate_tomato_image(image)
        
        if not validation_result["is_valid"]:
            logger.warning(f"Image validation failed: {validation_result['reason']}")
            return {
                "success": False,
                "message": "Please upload a clear image of tomato plant leaves",
                "suggestion": "Ensure the image contains visible green plant material and clear leaf details"
            }
        
        # Log validation success (internal only)
        logger.info(f"Image validation passed: {validation_result['reason']}")
        
        # Continue with normal prediction if validation passes
        processed_input = self.preprocess(image)
        raw_output = self.predict(processed_input)
        result = self.postprocess(raw_output)
        
        # Add success flag
        result["success"] = True
        
        return result
    
    def validate_tomato_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Internal validation to check if image likely contains plant material
        Returns: {"is_valid": bool, "reason": str}
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
                    "reason": f"Insufficient green content ({green_percentage:.1%}). Expected plant material."
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
                "reason": f"Image too uniform (texture score: {texture_score:.1f}). Expected detailed leaf structure."
            }
        
        # 3. SIZE AND ASPECT RATIO CHECK
        height, width = img_array.shape[:2]
        if min(height, width) < 100:
            return {
                "is_valid": False,
                "reason": f"Image too small ({width}x{height}). Minimum 100x100 pixels required."
            }
        
        # If all checks pass
        return {
            "is_valid": True,
            "reason": "Image passes pre-validation checks"
        }

# Instantiate the Predictor with TFLite model and Keras fallback
prediction_service = Predictor(TFLITE_MODEL_PATH, KERAS_MODEL_PATH)