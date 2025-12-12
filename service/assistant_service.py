import os
import logging
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")  

class AgriAssistant:
    """Agricultural AI assistant using Gemini model."""
    def __init__(self):
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY environment variable not set. AI assistant will be disabled.")
            self.model = None
            return
        
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(model_name=MODEL_NAME)
            logger.info(f"Gemini assistant initialized successfully with model: {MODEL_NAME}")  # Changed from logger.success

        except Exception as e:
            logger.error(f"Failed to initialize Gemini assistant: {e}")
            self.model = None

    def is_active(self) -> bool:
        """Check if the AI assistant is active and ready to use."""
        return self.model is not None

    def get_advice(self, disease: str, confidence: float, region: Optional[str] = None, language: str = "en") -> str:
        """Get agricultural advice based on detected disease and confidence level."""
        if not self.is_active():  # Use is_active() method
            logger.warning("AI assistant called but not active")
            return "AI assistant is currently unavailable. Please check API key configuration."
        
        # # Clean disease name for prompt
        # disease_name = disease.replace("Tomato___", "").replace("_", " ")

        # Construct prompt
        if "Healthy Plant" in disease.lower():
            prompt = (
                f"A tomato plant is healthy (confidence: {confidence:.1%}). "
                f"Give 3 brief tips to maintain this plant's health."
            )
        else:
            prompt = (
                f"A tomato plant is affected by {disease} (confidence: {confidence:.1%}). "
                f"Provide concise advice for a farmer:\n"
                f"1. Immediate Action\n"
                f"2. Organic Solution\n"
                f"3. Prevention"
            )
        
        if region:
            prompt += f"\nThe farm is located in {region}. Tailor the advice to this region."

        if language and language != "en":
            prompt += f"\nProvide the advice in {language}."

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            # return "Could not generate advice at this time. Please try again later."

# Create Agricultural assistant instance
agri_assistant = AgriAssistant()