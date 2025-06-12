# Usamos una imagen base oficial de Python
FROM python:3.11-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos requirements.txt y lo instalamos
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código fuente al contenedor
COPY ./app ./app

# Copiamos la carpeta templates al contenedor
COPY ./app/templates /app/templates

# Exponemos el puerto que usará la app
EXPOSE 8000

# Comando para correr la aplicación (FastAPI + Uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]