import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import os
from dotenv import load_dotenv

class LLM():
    def __init__(self) -> None:
        pass
    def model(self, message):
        load_dotenv()
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        response = model.generate_content([message], safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        })
        return response.text