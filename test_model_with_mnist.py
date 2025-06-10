import os
import onnxruntime as ort
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
import urllib.request
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
MODEL_URL = os.getenv("MODEL_URL")

# Descargar el modelo si no existe
if not os.path.exists("model.onnx"):
    print("Descargando modelo ONNX...")
    urllib.request.urlretrieve(MODEL_URL, "model.onnx")
    print("Modelo descargado.")

# Preprocesamiento
def preprocess(image_array):
    img = Image.fromarray(image_array).convert("L").resize((28, 28))
    img_arr = np.array(img).astype(np.float32) 
    img_arr /= 255.0
    return img_arr.reshape(1, 1, 28, 28)

# Cargar modelo
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
shape = session.get_inputs()[0].shape
type = session.get_inputs()[0].type


# Cargar datos de prueba
(_, _), (x_test, y_test) = mnist.load_data()

# Evaluar primeras 100 imágenes
correct = 0
for i in range(100):
    input_tensor = preprocess(x_test[i])
    pred = session.run(None, {input_name: input_tensor})
    pred_label = int(np.argmax(pred[0]))
    print(f"shape: {shape}, type: {type}")
    if pred_label == y_test[i]:
        correct += 1
    print(f"Imagen {i}: Real = {y_test[i]} | Predicho = {pred_label}")

print(f"\nPrecisión: {correct}/100 = {correct}%")
