#!/bin/bash

# Salir en caso de error
set -e

# Definir el directorio del proyecto (donde está el script)
PROJECT_DIR="$(pwd)"

# Nombre del contenedor
CONTAINER_NAME="azure_functions_pipeline"

echo "🚀 Construyendo imagen Docker..."
sudo docker build -t azure_functions-pipeline .

echo "🔄 Iniciando contenedor con volúmenes montados..."
sudo docker run --rm -it -p 7071:7071 \
    -v "$PROJECT_DIR/extracted_data:/home/site/wwwroot/extracted_data" \
    -v "$PROJECT_DIR/engineered_data:/home/site/wwwroot/engineered_data" \
    -v "$PROJECT_DIR/ml_results:/home/site/wwwroot/ml_results" \
    -v "$PROJECT_DIR:/home/site/wwwroot" \
    -e BASE_DIR="/home/site/wwwroot" \
    -w /home/site/wwwroot \
    --name $CONTAINER_NAME \
    azure_functions-pipeline \
    bash -c "

    echo '✅ Activando entorno virtual dentro del contenedor...'
    python -m venv env
    source env/bin/activate

    echo '📦 Instalando dependencias...'
    pip install --no-cache-dir -r requirements.txt

    echo '🧪 Ejecutando tests unitarios con cobertura...'
    pytest --cov=azure_func_data_processing --cov-report=html || echo '⚠️ Test unitarios fallaron'

    echo '📋 Ejecutando análisis de código con flake8...'
    flake8 azure_func_data_processing Cleaning Model tests --output-file lint_report.txt || echo '⚠️ Errores de linting'

    # 📂 Ir a la carpeta de funciones antes de iniciar Azure Functions
    echo '📂 Moviéndonos a azure_func_data_processing...'
    cd azure_func_data_processing

    echo '🚀 Iniciando Azure Functions en segundo plano...'
    func start &

    echo '✅ Azure Functions corriendo en el contenedor.'
    echo '👉 Ahora puedes ejecutar los curl desde otra terminal.'
    echo 'Ejemplo:'
    echo 'curl -X POST -H \"Content-Type: application/json\" -d \"{\\\"kaggle_competition\\\": \\\"ds-programming-test\\\"}\" http://localhost:7071/api/DataProcessingFunction'
    echo ''
    echo '⌛ El contenedor sigue corriendo para que puedas probar la API.'
    echo 'Para detenerlo, presiona Ctrl+C en esta terminal.'

    sleep infinity
"
