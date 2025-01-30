#!/bin/bash

# Salir en caso de error
set -e

echo "Activating Python virtual environment..."
python -m venv env
source env/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running unit tests with coverage..."
pytest --cov=azure_func_data_processing tests/ --cov-report=html

echo "Running linting..."
flake8 azure_func_data_processing Cleaning Model tests --output-file lint_report.txt

echo "Starting Azure Functions locally..."
cd azure_func_data_processing
func start &

# Dar tiempo a que las funciones se levanten
sleep 10

echo "Triggering Azure Functions..."
curl -X POST -H "Content-Type: application/json" -d '{
    "kaggle_competition": "ds-programming-test"
}' http://localhost:7071/api/DataProcessingFunction

curl -X POST -H "Content-Type: application/json" -d '{}' http://localhost:7071/api/MLPipelineFunction

# Finalizar Azure Functions
echo "Stopping Azure Functions..."
pkill -f 'func start'

echo "Pipeline executed successfully."
