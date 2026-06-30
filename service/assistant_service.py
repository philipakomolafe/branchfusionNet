import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# Use the new google.genai package
from google import genai


class AgriAssistant:
    """Agricultural AI assistant using Gemini model."""

    def __init__(self):
        if not GEMINI_API_KEY:
            logger.warning(
                "GEMINI_API_KEY environment variable not set. "
                "AI assistant will be disabled."
            )
            self.client = None
            return
        try:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"Gemini assistant initialized successfully with model: {MODEL_NAME}")
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
                f"\nProvide your ENTIRE response in {full_lang} language only. "
                f"Do not include any English words or sentences."
            )

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            # Safely verify the response actually contains usable text
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
            else:
                logger.error(f"Gemini returned an empty response object: {response}")
                return "Could not generate advice. The AI returned an empty response."

        except Exception as e:
            # Log the full traceback so the real cause shows up in Render logs
            logger.error(f"Gemini generation error: {str(e)}", exc_info=True)
            # Temporarily surface the real error in the API response itself,
            # so it's visible even without checking server logs.
            # Remove the str(e) part once the root cause is confirmed and fixed.
            return f"Could not generate advice due to an internal error: {str(e)}"


agri_assistant = AgriAssistant()