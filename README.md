# ONNX Deployment System 

Este proyecto implementa un sistema de despliegue automático para modelos ONNX usando FastAPI, Docker y GitHub Actions. Utiliza el modelo preentrenado `mnist-12.onnx` para reconocimiento de dígitos escritos a mano. Está diseñado para facilitar la actualización y publicación de nuevos modelos en producción.

##  Características

-  Despliegue automático a entornos `dev` y `prod` mediante CI/CD.
-  Modelo preentrenado en formato ONNX para clasificación de dígitos.
-  Pipeline con pruebas automáticas y control de calidad.
-  Contenedor Docker listo para ser desplegado en Azure Web App for Containers.
-  Aplicación con API REST y frontend básico para predicción.

---

##  Modelo utilizado: `mnist-12.onnx`

Este proyecto utiliza un modelo preentrenado en formato ONNX llamado `mnist-12.onnx`, descargado directamente desde el repositorio oficial de modelos de ONNX: [https://github.com/onnx/models/tree/main/validated/vision/classification/mnist).

###  ¿Qué es MNIST?

MNIST (Modified National Institute of Standards and Technology) es un dataset ampliamente utilizado para entrenar y evaluar modelos de reconocimiento de dígitos manuscritos. Contiene:

- **60,000 imágenes** para entrenamiento.
- **10,000 imágenes** para pruebas.
- Imágenes en **escala de grises**, con tamaño **28x28 píxeles**.
- Cada imagen representa un solo **dígito del 0 al 9**.

###  Detalles del modelo `mnist-12.onnx`

El modelo está basado en una red neuronal convolucional (CNN), diseñada para reconocer dígitos a partir de imágenes procesadas. Fue entrenado previamente con un framework como PyTorch o TensorFlow y convertido al formato ONNX, que permite su uso multiplataforma.

###  Entrada y salida del modelo

- **Entrada esperada**: tensor de tamaño `(1, 1, 28, 28)` que representa una imagen en blanco y negro de un dígito.
- **Salida**: vector con 10 probabilidades, una por cada clase (dígito). La clase con mayor probabilidad es la predicción del modelo.

###  Ventajas para este proyecto

- Modelo **liviano**, eficiente y rápido.
- Totalmente **portable** y compatible con `onnxruntime`.
- Ideal para **demostrar despliegue automático**, pruebas unitarias y predicciones en tiempo real desde una interfaz web.

---

##  Instrucciones para ejecutar la API con FastAPI y Templates

### Requisitos

- Python 3.9+
- `pip`
- Docker (opcional para despliegue)
- `virtualenv` (opcional pero recomendado)

### 1.  Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/onnx-deployment-system.git
cd onnx-deployment-system
```
### 2. Crear entorno virtual e instalar dependencias


```bash
python -m venv env
source env/bin/activate  # En Linux
.\env\Scripts\activate #En Windows
pip install -r requirements.txt

```

### 3. Descargar el modelo ONNX

El modelo no está en el repositorio. Debes definir su URL en una variable de entorno '.env':

```bash
MODEL_URL = ""https://url-del-modelo/mnist-12.onnx""
```

### 4. Ejecutar la aplicación FastAPI

```bash
uvicorn app.main:app --reload
```
La app se ejecutará en: http://127.0.0.1:8000


### Alternativa: Ejecutar en docker

```bash
docker build -t onnx-api-test .
docker run -p 8000:8000 --env-file .env onnx-api-test
```
