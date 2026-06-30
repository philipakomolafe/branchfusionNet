import os
import logging
from typing import Optional
from dotenv import load_dotenv

# 1. Safely load local .env if it exists. 
# If it's missing (on your cloud server), it skips this without breaking.
if os.path.exists(".env"):
    load_dotenv()

logger = logging.getLogger(__name__)

# 2. Extract configuration variables from the system environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# Import the correct Google GenAI library
from google import genai


class AgriAssistant:
    """Agricultural AI assistant using Gemini model."""

    def __init__(self):
        # 3. Explicitly verify that the key string actually exists
        if not GEMINI_API_KEY:
            logger.warning(
                "GEMINI_API_KEY environment variable not detected by host server. "
                "AI assistant fallback triggered."
            )
            self.client = None
            return
            
        try:
            # 4. Explicitly bind the variable string into the Client initialization
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"Gemini assistant initialized successfully on host with model: {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini assistant: {e}")
            self.client = None

    def is_active(self) -> bool:
        return self.client is not None

    def get_advice(
        self,
        disease: str,
        confidence: float,
        region: Optional[str] = None,
        language: str = "en",
    ) -> str:
        if not self.is_active():
            logger.warning("AI assistant called but not active")
            return "AI assistant is currently unavailable. Please check API key configuration."

        disease_name = disease.replace("Tomato___", "").replace("_", " ")

        if "healthy" in disease_name.lower():
            prompt = (
                f"A tomato plant is healthy (confidence: {confidence:.1%}). "
                f"Give 3 brief tips to maintain this plant's health."
            )
        else:
            prompt = (
                f"A tomato plant is affected by {disease_name} "
                f"(confidence: {confidence:.1%}). "
                f"Provide concise advice for a farmer:\n"
                f"1. Immediate Action\n"
                f"2. Organic Solution\n"
                f"3. Prevention"
            )

        if region:
            prompt += f"\nThe farm is located in {region}. Tailor the advice to this region."

        # Explicit language instruction — critical for Yoruba/Hausa
        lang_map = {"yo": "Yoruba", "ha": "Hausa", "en": "English"}
        full_lang = lang_map.get(language, language)

        if full_lang != "English":
            prompt += (
                f"\nTranslate the advice accurately into {full_lang}. "
                f"Use regional agricultural terminology where appropriate."
            )

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
            else:
                logger.error(f"Gemini returned an empty response object: {response}")
                return "Could not generate advice. The AI returned an empty response."

        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}", exc_info=True)
            return "Could not generate advice at this time. Please try again later."


agri_assistant = AgriAssistant()
