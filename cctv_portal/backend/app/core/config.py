import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    YOLO_MODEL = "yolo11n.pt"
    PROCESS_FPS = 5
    
    # Bucket names
    EVIDENCE_BUCKET = "evidence"

settings = Config()
