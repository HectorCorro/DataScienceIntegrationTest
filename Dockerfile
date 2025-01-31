# Imagen base de Azure Functions con Python 3.12
FROM mcr.microsoft.com/azure-functions/python:4-python3.12

# Establecer directorio de trabajo
WORKDIR /home/site/wwwroot

# Instalar Azure Functions Core Tools en Debian
RUN apt-get update && \
    apt-get install -y curl gpg ca-certificates && \
    curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg && \
    echo "deb [arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/microsoft.list && \
    apt-get update && \
    apt-get install -y azure-functions-core-tools-4 && \
    apt-get clean

# Verificar instalación
RUN func --version

COPY requirements.txt .

# Instalar dependencias globales en el entorno principal
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar el archivo de dependencias específico para Azure Functions
COPY azure_func_data_processing/requirements.txt azure_func_data_processing/

# Instalar dependencias en el entorno de Azure Functions
RUN python -m pip install --upgrade pip && \
    python -m pip install --target="/home/site/wwwroot/.python_packages/lib/site-packages" -r azure_func_data_processing/requirements.txt

# Crear el directorio de configuración de Kaggle y copiar el archivo de credenciales
RUN mkdir -p /home/.config/kaggle
COPY kaggle.json /home/.config/kaggle/kaggle.json

# Establecer permisos adecuados
RUN chmod 600 /home/.config/kaggle/kaggle.json

# 🔹 Definir variable de entorno para la ruta base
ENV BASE_DIR="/home/site/wwwroot"

# Copiar todo el código del proyecto al contenedor
COPY . .

# Establecer PYTHONPATH para incluir la carpeta de funciones
ENV PYTHONPATH="/home/site/wwwroot/azure_func_data_processing:/home/site/wwwroot/.python_packages/lib/site-packages"
RUN chmod +x /home/site/wwwroot/run_pipeline.sh

# Exponer puerto para Azure Functions
EXPOSE 7071

# Comando para iniciar pruebas de integración del pipeline
CMD ["func", "start", "--verbose", "--script-root", "/home/site/wwwroot/azure_func_data_processing"]