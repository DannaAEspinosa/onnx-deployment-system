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

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ruta y URL del modelo
MODEL_PATH = "mnist12.onnx"
MODEL_URL = os.getenv("MODEL_URL")

# Descargar el modelo si no existe
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde:", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Cargar modelo ONNX
session = ort.InferenceSession(MODEL_PATH)

class PredictRequest(BaseModel):
    image_data: str  # Base64 string del dibujo del usuario

def preprocess(image_bytes):
    # Preprocesa la imagen para el modelo MNIST: escala 28x28, escala 0-1
    img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
    img_arr = np.array(img).astype(np.float32)
    img_arr = 255 - img_arr  # Invertir colores si el fondo es blanco
    img_arr /= 255.0
    img_arr = img_arr.reshape(28, 28)  # para guardar como imagen
    Image.fromarray((img_arr * 255).astype(np.uint8)).save("debug_preprocessed.png")
    img_arr = img_arr.reshape(1, 1, 28, 28)
    return img_arr

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: PredictRequest):
    # El cliente envía la imagen base64 (png) del canvas
    header, encoded = request.image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    input_tensor = preprocess(image_bytes)
    input_name = session.get_inputs()[0].name
    pred_onx = session.run(None, {input_name: input_tensor})
    predicted_number = int(np.argmax(pred_onx[0]))

    return JSONResponse({"prediction": predicted_number})
