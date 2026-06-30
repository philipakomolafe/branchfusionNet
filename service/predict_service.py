from typing import Any, Dict
import logging
import pathlib
import numpy as np
from PIL import Image
import warnings

logger = logging.getLogger(__name__)

TFLITE_MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" / "MultiBranchFusionNet-0.tflite"
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

disease_display_names_yo = {
    "Grey_leaf_spot_(fungi)": "Abàwọn Ewé Grẹy (Elu)",
    "Tomato___Bacterial_spot": "Abàwọn Bakitéríà",
    "Tomato___Early_blight": "Ìbàjẹ́ Ìbẹ̀rẹ̀",
    "Tomato___Late_blight": "Ìbàjẹ́ Ìkẹhìn",
    "Tomato___Leaf_Mold": "Elu Ewé",
    "Tomato___Septoria_leaf_spot": "Abàwọn Ewé Septoria",
    "Tomato___Spider_mites": "Àkàlàmọ̀ Spider",
    "Tomato___Target_Spot": "Abàwọn Ibi-àfojúsùn",
    "Tomato___Yellow_Leaf_Curl_Virus": "Fáírọ́ọ̀sì Yíyípo Ewé",
    "Tomato___healthy": "Ọgbìn Tó Ní Ìlera"
}

disease_treatments_yo = {
    "Grey_leaf_spot_(fungi)": "Yọ àwọn ewé tó ní àrùn kúrò, mú afẹ́fẹ́ ṣàn dáadáa, kí o sì lo oogun egbòogi tó ní chlorothalonil tàbí bàbà.",
    "Tomato___Bacterial_spot": "Yọ àwọn apá ọgbìn tó ní àkóràn kúrò, lo oogun bactericide tó ní bàbà, kí o sì yẹra fún omi ìsanra lórí ewé.",
    "Tomato___Early_blight": "Yọ àwọn ewé ìsàlẹ̀ tó ní àkóràn kúrò, lo oogun egbòogi, kí o sì fi mulch ká ọgbìn láti dènà omi ilẹ̀.",
    "Tomato___Late_blight": "Yọ àti run àwọn ọgbìn tó ní àkóràn LẸSẸKẸSẸ. Lo oogun egbòogi systemic ní kíákíá kí o sì yẹ agbègbè tó níyọ sọ́tọ̀.",
    "Tomato___Leaf_Mold": "Mú afẹ́fẹ́ ṣàn dáadáa, dínkù ọ̀rọ̀ ojú ọjọ́ ní àyíká ọgbìn, kí o sì lo oogun egbòogi.",
    "Tomato___Septoria_leaf_spot": "Yọ àwọn ewé ìsàlẹ̀ tó ní àkóràn kúrò, lo oogun egbòogi, kí o sì gbé omi sí ìpele ilẹ̀ nìkan.",
    "Tomato___Spider_mites": "Fọ ọgbìn pẹ̀lú omi láti lé mites kúrò, lo ọxẹ̀ insecticidal, kí o sì mú ọ̀rọ̀ ojú ọjọ́ pọ̀ sí i.",
    "Tomato___Target_Spot": "Yọ àwọn ewé tó ní àkóràn kúrò, lo oogun egbòogi, kí o sì mú afẹ́fẹ́ ṣàn dáadáa.",
    "Tomato___Yellow_Leaf_Curl_Virus": "Yọ àwọn ọgbìn tó ní àkóràn kúrò lẹsẹkẹsẹ, ṣàkóso àwọn whitefly, kí o sì lo oríṣi tomati tó tako fáírọ́ọ̀sì.",
    "Tomato___healthy": "Tẹ̀síwájú ìtọ́jú lọwọlọwọ. Ṣe àbójútó déédéé fún àmì àrùn ní àkókò ìbẹ̀rẹ̀ kí o sì ṣọ́ ọgbà rẹ dáadáa."
}

disease_display_names_ha = {
    "Grey_leaf_spot_(fungi)": "Tabo Mai Launin Toka (Namomin Kaza)",
    "Tomato___Bacterial_spot": "Tabo na Ƙwayoyin Cuta",
    "Tomato___Early_blight": "Lalacewar Farko",
    "Tomato___Late_blight": "Lalacewar Ƙarshe",
    "Tomato___Leaf_Mold": "Namomin Kaza na Ganye",
    "Tomato___Septoria_leaf_spot": "Tabo na Ganye Septoria",
    "Tomato___Spider_mites": "Ƙwaro Spider",
    "Tomato___Target_Spot": "Tabo Mai Nishani",
    "Tomato___Yellow_Leaf_Curl_Virus": "Cutar Juɓewar Ganye Mai Rawaya",
    "Tomato___healthy": "Shuka Mai Lafiya"
}

disease_treatments_ha = {
    "Grey_leaf_spot_(fungi)": "Cire ganyen da suka kamu, inganta zirga-zirgar iska, kuma yi amfani da maganin fungicide mai chlorothalonil ko jan ƙarfe.",
    "Tomato___Bacterial_spot": "Cire sassan shuka da suka kamu, yi amfani da maganin bactericide mai jan ƙarfe, kuma guji ban ruwa daga sama.",
    "Tomato___Early_blight": "Cire ganyen da suka kamu na ƙasa, yi amfani da fungicide, kuma zuba mulch a kusa da shuke-shuke don hana ruwan ƙasa.",
    "Tomato___Late_blight": "Cire kuma lalata shuke-shuken da suka kamu YANZU HAKA. Yi amfani da systemic fungicide da gaggawa kuma keɓe yankin da ya kamu.",
    "Tomato___Leaf_Mold": "Inganta zirga-zirgar iska, rage zafi da danshi a kusa da shuke-shuke, kuma yi amfani da maganin fungicide.",
    "Tomato___Septoria_leaf_spot": "Cire ganyen da suka kamu na ƙasa, yi amfani da fungicide, kuma shayar da ruwa a matakin ƙasa ne kawai.",
    "Tomato___Spider_mites": "Fesa ruwa a kan shuke-shuke don kawar da mites, yi amfani da sabulu na insecticidal, kuma ƙara danshi.",
    "Tomato___Target_Spot": "Cire ganyen da suka kamu, yi amfani da fungicide, kuma inganta zirga-zirgar iska.",
    "Tomato___Yellow_Leaf_Curl_Virus": "Cire shuke-shuken da suka kamu nan da nan, sarrafa ƙudajen farin whitefly, kuma yi amfani da nau'in tomato mai tsayin daka ga cutar.",
    "Tomato___healthy": "Ci gaba da aikin kula na yanzu. Duba akai-akai don alamun cuta a farkon lokaci kuma kiyaye tsabtar lambun."
}

def get_display_name(disease: str, language: str) -> str:
    if language == "yo":
        return disease_display_names_yo.get(disease, disease_display_names.get(disease, disease))
    if language == "ha":
        return disease_display_names_ha.get(disease, disease_display_names.get(disease, disease))
    return disease_display_names.get(disease, disease)


def get_treatment(disease: str, language: str) -> str:
    if language == "yo":
        return disease_treatments_yo.get(disease, disease_treatments.get(disease, ""))
    if language == "ha":
        return disease_treatments_ha.get(disease, disease_treatments.get(disease, ""))
    return disease_treatments.get(disease, "")


def _load_tflite_interpreter(model_path: pathlib.Path):
    """Load a TFLite model using LiteRT only - no TensorFlow dependency."""
    import ai_edge_litert.interpreter as litert_interpreter
    interpreter = litert_interpreter.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


class TomatoValidator:
    """Validator class to check if image contains a tomato plant. Uses standalone Keras (not TensorFlow)."""
    def __init__(self, model_path: pathlib.Path):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.confidence_threshold = 0.7

    def load_model(self) -> None:
        if not self.model_path.exists():
            logger.warning(f"Validator model not found at {self.model_path}. Validation will be skipped.")
            return
        try:
            import keras
            self.model = keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Tomato validator model loaded from {self.model_path}")
            logger.info(f"Validator input shape: {self.model.input_shape}, output shape: {self.model.output_shape}")
        except Exception as e:
            logger.error(f"Failed to load validator model: {e}")
            self.is_loaded = False

    def preprocess(self, image: Image.Image) -> np.ndarray:
        target_size = (224, 224)
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)

    def validate(self, image: Image.Image) -> Dict[str, Any]:
        if not self.is_loaded:
            logger.warning("Validator model not loaded. Skipping tomato validation.")
            return {"is_tomato": True, "confidence": 1.0, "message": "Validation skipped (model not available)"}

        try:
            processed = self.preprocess(image)
            prediction = self.model.predict(processed, verbose=0)

            tomato_confidence = float(prediction[0][0])
            is_tomato = tomato_confidence >= self.confidence_threshold

            message = (
                f"Image verified as tomato plant with ({tomato_confidence:.1%} confidence score)"
                if is_tomato else
                f"Image does not appear to be a tomato plant with ({tomato_confidence:.1%} confidence score)"
            )
            return {"is_tomato": is_tomato, "confidence": tomato_confidence, "message": message}

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"is_tomato": True, "confidence": 0.0, "message": f"Validation error: {str(e)}. Proceeding with prediction."}


class Predictor:
    """Predictor class for tomato plant disease classification using TFLite only."""

    def __init__(self, tflite_model_path: pathlib.Path, validator_model_path: pathlib.Path = None):
        self.tflite_model_path = tflite_model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_type = None
        self.validator = TomatoValidator(validator_model_path) if validator_model_path else None

    def load_model(self) -> None:
        try:
            if self.validator:
                self.validator.load_model()

            if not self.tflite_model_path.exists():
                raise FileNotFoundError(f"TFLite model not found: {self.tflite_model_path}")

            self.interpreter = _load_tflite_interpreter(self.tflite_model_path)
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_type = "tflite"
            logger.info(f"TFLite model loaded successfully from {self.tflite_model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess(self, image: Image.Image) -> np.ndarray:
        input_shape = self.input_details[0]['shape']
        target_size = (input_shape[1], input_shape[2])
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)

    def predict(self, processed_input: np.ndarray) -> np.ndarray:
        if self.interpreter is None:
            raise RuntimeError("TFLite model not loaded. Call load_model() first.")
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_input)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def postprocess(self, raw_output: np.ndarray) -> Dict[str, Any]:
        probabilities = raw_output[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        return {"disease": predicted_class, "confidence": confidence}

    def predict_from_image(self, image: Image.Image) -> Dict[str, Any]:
        if self.model_type is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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

        quality_check = self._validate_image_quality(image)
        if not quality_check["is_valid"]:
            return {
                "success": False,
                "message": quality_check["reason"],
                "suggestion": "Ensure the image has good lighting and clear leaf details"
            }

        processed_input = self.preprocess(image)
        raw_output = self.predict(processed_input)
        result = self.postprocess(raw_output)
        result["success"] = True
        return result

    def _validate_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Basic image quality checks using PIL only - no OpenCV/TensorFlow needed."""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        if min(height, width) < 100:
            return {"is_valid": False, "reason": f"Image too small ({width}x{height}). Minimum 100x100 pixels required."}

        # Green-pixel check using simple RGB thresholding instead of cv2.cvtColor
        if len(img_array.shape) == 3:
            r = img_array[:, :, 0].astype(np.int32)
            g = img_array[:, :, 1].astype(np.int32)
            b = img_array[:, :, 2].astype(np.int32)
            # crude "is this pixel green-ish" check: green channel dominates red and blue
            green_mask = (g > r) & (g > b) & (g > 40)
            green_percentage = np.sum(green_mask) / green_mask.size

            if green_percentage < 0.10:
                return {"is_valid": False, "reason": f"Image lacks sufficient green plant material ({green_percentage:.1%})"}

        return {"is_valid": True, "reason": "Image quality acceptable"}


# Instantiate the Predictor - disease model is TFLite, validator is standalone Keras
prediction_service = Predictor(TFLITE_MODEL_PATH, VALIDATOR_MODEL_PATH)
