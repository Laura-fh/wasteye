# schemas.py

# 3º schemas.py
#JSONM struture
from pydantic import BaseModel

class ImageRequest(BaseModel):
    image: str  # base64 encoded image must be str
