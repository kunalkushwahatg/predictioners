from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi.responses import HTMLResponse

from pydantic import BaseModel
from fastapi import HTTPException

import os
import joblib
import numpy as np



app = FastAPI()


# Setup templates
templates = Jinja2Templates(directory="templates")

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the directory where models are stored
MODELS_DIR = "saved_models"

# Pydantic model for request validation
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    model_type: str  # 'deaths' or 'cfr'
    model_name: str  # e.g., 'linear_reg', 'random_forest', etc.

# Helper function to load model and scaler
def load_model_and_scaler(model_type: str, model_name: str):
    """
    Load the model and its corresponding scaler from disk.
    """
    model_path = os.path.join(MODELS_DIR, model_type, f"{model_name}.pkl")
    scaler_path = os.path.join(MODELS_DIR, model_type, f"{model_name}_scaler.pkl")
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found for {model_type}/{model_name}")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Endpo  to get available models
@app.get("/models")
async def get_available_models():
    """
    Returns a list of available models for deaths and CFR predictions.
    """
    models = {
        "deaths": [],
        "cfr": []
    }
    for model_type in models.keys():
        model_dir = os.path.join(MODELS_DIR, model_type)
        if os.path.exists(model_dir):
            # List all model files (excluding scalers)
            models[model_type] = [
                f.replace('.pkl', '') for f in os.listdir(model_dir)
                if f.endswith('.pkl') and '_scaler' not in f
            ]
    return models

# Endpoint to make predictions
@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Makes a prediction using the selected model.
    """
    try:
        # Load the model and scaler
        model, scaler = load_model_and_scaler(request.model_type, request.model_name)
        
        # Prepare input data
        input_data = np.array([[request.latitude, request.longitude]])
        
        # Standardize the input
        scaled_data = scaler.transform(input_data)


        # Make prediction
        prediction = model.predict(scaled_data)
        
        # Return the prediction
        return {"prediction": float(prediction[0])}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Serve the frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the frontend HTML file.
    """
    with open("templates/index.html", "r") as file:
        return HTMLResponse(content=file.read())

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)