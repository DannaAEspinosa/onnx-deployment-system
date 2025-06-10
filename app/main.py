from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import base64
from PIL import Image
import io
import os
import urllib.request
import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobClient

# Cargar variables de entorno
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Variables de entorno
MODEL_PATH = "mnist12.onnx"
MODEL_URL = os.getenv("MODEL_URL")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde:", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Cargar modelo ONNX
session = ort.InferenceSession(MODEL_PATH)

class PredictRequest(BaseModel):
    image_data: str  # Base64 string del dibujo del usuario

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
    img_arr = np.array(img).astype(np.float32)
    img_arr /= 255.0
    img_arr = img_arr.reshape(28, 28)  # solo para debug opcional
    Image.fromarray((img_arr * 255).astype(np.uint8)).save("debug_preprocessed.png")
    img_arr = img_arr.reshape(1, 1, 28, 28)
    return img_arr

def log_prediction(prediction: int):
    filename = f"predicciones_{ENVIRONMENT}.txt"
    log_line = f"{datetime.datetime.now()}: {prediction}\n"

    # Guardar en archivo local
    with open(filename, "a") as f:
        f.write(log_line)

    # Subir a Azure Blob
    try:
        blob_client = BlobClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING,
            container_name=AZURE_STORAGE_CONTAINER,
            blob_name=filename
        )
        with open(filename, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"[INFO] Archivo subido a Azure: {filename}")
    except Exception as e:
        print(f"[ERROR] Fallo al subir el archivo: {e}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        header, encoded = request.image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        input_tensor = preprocess(image_bytes)
        input_name = session.get_inputs()[0].name
        pred_onx = session.run(None, {input_name: input_tensor})
        predicted_number = int(np.argmax(pred_onx[0]))

        # Log y subir a Azure
        log_prediction(predicted_number)

        return JSONResponse({"prediction": predicted_number})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
