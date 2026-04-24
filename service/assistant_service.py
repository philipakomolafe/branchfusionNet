import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Prefer the new `google.genai` package, fall back to deprecated `google.generativeai` if needed
genai_pkg = None
try:
    import google.genai as genai_pkg
    logger.info("Using google.genai package for AI assistant")
except Exception:
    try:
        import google.generativeai as genai_pkg
        logger.warning("Using deprecated google.generativeai package; please migrate to google.genai")
    except Exception:
        genai_pkg = None


class AgriAssistant:
    """Agricultural AI assistant (supports google.genai and google.generativeai)."""

    def __init__(self):
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY environment variable not set. AI assistant will be disabled.")
            self.client = None
            self.model = None
            return

        self.client = None
        self.model = None

        if genai_pkg is None:
            logger.warning("No GenAI package available (google.genai or google.generativeai). Assistant disabled.")
            return

        try:
            if hasattr(genai_pkg, "configure"):
                try:
                    genai_pkg.configure(api_key=GEMINI_API_KEY)
                except Exception:
                    pass

            if hasattr(genai_pkg, "Client"):
                try:
                    self.client = genai_pkg.Client(api_key=GEMINI_API_KEY)
                except Exception:
                    try:
                        self.client = genai_pkg.Client()
                    except Exception:
                        self.client = genai_pkg
            elif hasattr(genai_pkg, "TextGenerationClient"):
                try:
                    self.client = genai_pkg.TextGenerationClient()
                except Exception:
                    self.client = genai_pkg
            elif hasattr(genai_pkg, "GenerativeModel"):
                try:
                    self.model = genai_pkg.GenerativeModel(model_name=MODEL_NAME)
                except Exception:
                    self.model = None
            else:
                self.client = genai_pkg

            logger.info(f"Gemini assistant initialized with model: {MODEL_NAME}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini assistant: {e}")
            self.client = None
            self.model = None

    def is_active(self) -> bool:
        return self.client is not None or self.model is not None

    def _extract_text(self, response) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        try:
            if hasattr(response, "text"):
                t = getattr(response, "text")
                if isinstance(t, str):
                    return t
            if hasattr(response, "content"):
                c = getattr(response, "content")
                if isinstance(c, str):
                    return c
            if hasattr(response, "candidates"):
                candidates = getattr(response, "candidates")
                if candidates:
                    first = candidates[0]
                    if isinstance(first, dict):
                        for key in ("content", "text"):
                            if key in first:
                                val = first[key]
                                if isinstance(val, str):
                                    return val
                                if isinstance(val, list) and val:
                                    return val[0]
                    if hasattr(first, "text"):
                        return getattr(first, "text")
                    return str(first)
            if isinstance(response, dict):
                if "candidates" in response and response["candidates"]:
                    cand = response["candidates"][0]
                    if isinstance(cand, dict):
                        for key in ("content", "text"):
                            if key in cand:
                                val = cand[key]
                                if isinstance(val, str):
                                    return val
                                if isinstance(val, list) and val:
                                    return val[0]
                    return str(cand)
                if "output" in response and response["output"]:
                    out = response["output"][0]
                    if isinstance(out, dict) and "content" in out:
                        c = out["content"]
                        if isinstance(c, list) and c:
                            first = c[0]
                            if isinstance(first, dict) and "text" in first:
                                return first["text"]
                            if isinstance(first, str):
                                return first
                        if isinstance(c, str):
                            return c
            return str(response)
        except Exception:
            try:
                return str(response)
            except Exception:
                return ""

    def get_advice(self, disease: str, confidence: float, region: Optional[str] = None, language: str = "en") -> str:
        if not self.is_active():
            logger.warning("AI assistant called but not active")
            return "AI assistant is currently unavailable. Please check API key configuration."

        if "healthy plant" in disease.lower():
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
            if self.client:
                try:
                    if hasattr(self.client, "responses") and hasattr(self.client.responses, "generate"):
                        resp = self.client.responses.generate(model=MODEL_NAME, input=prompt)
                        return self._extract_text(resp)
                    if hasattr(self.client, "generate_text"):
                        resp = self.client.generate_text(model=MODEL_NAME, input=prompt)
                        return self._extract_text(resp)
                    if hasattr(self.client, "generate"):
                        try:
                            resp = self.client.generate(model=MODEL_NAME, prompt=prompt)
                        except TypeError:
                            resp = self.client.generate(prompt)
                        return self._extract_text(resp)
                    if hasattr(self.client, "generate_content"):
                        resp = self.client.generate_content(prompt)
                        return self._extract_text(resp)
                except Exception as e:
                    logger.error(f"GenAI client generation error: {e}")

            if self.model and hasattr(self.model, "generate_content"):
                try:
                    resp = self.model.generate_content(prompt)
                    return self._extract_text(resp)
                except Exception as e:
                    logger.error(f"GenAI model generation error: {e}")

        except Exception as e:
            logger.error(f"GenAI generation error: {e}")

        return "Could not generate advice at this time. Please try again later."


# Create Agricultural assistant instance
agri_assistant = AgriAssistant()