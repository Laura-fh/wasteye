# 4º main.py
# This version uses the model_loader.py (Option 1) where the path logic is internal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Assuming schemas.py contains:
# from pydantic import BaseModel
# class ImageRequest(BaseModel):
#     image: str
from schemas import ImageRequest
from model_loader import load_model # Imports the function from the file above
# Assuming utils.py contains the necessary functions:
# def decode_base64_image(base64_str): ...
# def encode_image_to_base64(image): ...
# def draw_boxes_on_image(img_array, results, names): ...
from utils import decode_base64_image, encode_image_to_base64, draw_boxes_on_image
import numpy as np
from PIL import Image
import logging # Import logging for better feedback than print

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS allows access from other apps
logger.info("Setting up CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (consider restricting in production)
    allow_credentials=True, # Be careful with this and allow_origins=["*"]
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)
logger.info("CORS middleware configured.")

# Load the model once at startup using the imported function
logger.info("Attempting to load model at startup...")
try:
    # The logic to find 'best.pt' is now inside load_model()
    app.state.model = load_model()
    if app.state.model is None:
         logger.error("Model loading returned None. API might not function correctly.")
         # Decide if the app should stop if the model didn't load
         # raise RuntimeError("Failed to load model during startup.")
    else:
        logger.info("Model loaded and assigned to app.state.model successfully.")
        # Check for model names attribute, crucial for drawing boxes
        if not hasattr(app.state.model, 'names'):
             logger.warning("Loaded model does not have a '.names' attribute for class labels!")

except Exception as e:
    logger.exception(f"FATAL: Failed to load model during startup: {e}")
    # Exit if the model cannot be loaded, as the predict endpoint depends on it
    # You might want a more graceful shutdown depending on the environment
    import sys
    sys.exit(f"Failed to load model: {e}")


@app.post("/predict")
def predict(request: ImageRequest):
    """
    Handles image prediction requests.
    (This is the original synchronous version)
    """
    request_id = os.urandom(4).hex() # Simple ID for tracking request
    logger.info(f"[Req {request_id}] Received /predict request.")

    # Check if model was loaded correctly
    if not hasattr(app.state, 'model') or app.state.model is None:
         logger.error(f"[Req {request_id}] Model not available in app state.")
         # Use HTTPException for API errors
         raise HTTPException(status_code=503, detail="Model is not loaded or unavailable")

    model = app.state.model

    try:
        logger.info(f"[Req {request_id}] Decoding image...")
        image = decode_base64_image(request.image)
        if image is None:
            logger.error(f"[Req {request_id}] Decoding failed, image is None.")
            raise HTTPException(status_code=400, detail="Invalid Base64 image data")

        img_array = np.array(image)
        logger.info(f"[Req {request_id}] Image decoded, shape: {img_array.shape}. Running inference...")

        # --- Inference ---
        results = model(img_array)[0] # Assuming results are in the first element
        logger.info(f"[Req {request_id}] Inference complete.")

        # --- Drawing Boxes ---
        # Check if model has 'names' before trying to use it
        model_names = getattr(model, 'names', None)
        if model_names is None:
            logger.warning(f"[Req {request_id}] Model has no 'names' attribute. Cannot draw labels on boxes.")
            # Decide how to proceed: maybe draw boxes without labels, or skip drawing
            # For now, we pass None, assuming draw_boxes_on_image can handle it
            # Or raise an error if names are essential:
            # raise HTTPException(status_code=500, detail="Model configuration error: Class names missing")

        logger.info(f"[Req {request_id}] Drawing boxes...")
        # Pass model_names (which might be None)
        img_array_with_boxes = draw_boxes_on_image(img_array, results, model_names)
        logger.info(f"[Req {request_id}] Boxes drawn.")

        # --- Encoding Output ---
        output_image = Image.fromarray(img_array_with_boxes)
        logger.info(f"[Req {request_id}] Encoding result image...")
        result_base64 = encode_image_to_base64(output_image)
        logger.info(f"[Req {request_id}] Encoding complete. Returning response.")

        return {"result_image": result_base64}

    except HTTPException as http_exc:
         # Re-raise HTTPExceptions directly
         raise http_exc
    except Exception as e:
        # Catch-all for other unexpected errors during prediction
        logger.exception(f"[Req {request_id}] Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


@app.get("/")
def root():
    """ Basic root endpoint. """
    logger.info("Root endpoint '/' accessed.")
    return {"greeting": "hello!"}


# Add Uvicorn runner for convenience if running directly
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    # Note: reload=True is for development, disable in production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)