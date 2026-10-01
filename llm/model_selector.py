import threading
import os
from dotenv import load_dotenv


MODEL_BATCH_SIZE = 15
_model_lock = threading.Lock()
_model_request_count = 0

load_dotenv()

def get_next_gemini_model_name() -> str:
    global _model_request_count

    models = [
        name.strip()
        for name in os.getenv("GOOGLE_GEMINI_MODELS", "").split(",")
        if name.strip()
    ]
    if not models:
        model = os.getenv("GOOGLE_GEMINI_MODEL")
        if not model:
            raise ValueError("Set GOOGLE_GEMINI_MODELS or GOOGLE_GEMINI_MODEL")
        return model

    with _model_lock:
        model_index = (_model_request_count // MODEL_BATCH_SIZE) % len(models)
        _model_request_count += 1
        return models[model_index]