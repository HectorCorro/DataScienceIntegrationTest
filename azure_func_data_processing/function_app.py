import azure.functions as func
import pandas as pd
import requests
import datetime
import json
import logging
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

app = func.FunctionApp()

@app.route(route="DataProcessingFunction", auth_level=func.AuthLevel.ANONYMOUS)
def DataProcessingFunction(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Leer parámetros del POST request
        req_body = req.get_json()
        kaggle_competition = req_body.get("kaggle_competition")  # Nombre del concurso de Kaggle
        if not kaggle_competition:
            return func.HttpResponse(
                "Missing 'kaggle_competition' parameter in request body.",
                status_code=400
            )

        # Configurar rutas
        repo_directory = os.path.dirname(os.path.abspath(__file__))  # Directorio actual
        parent_directory = os.path.dirname(repo_directory)  # Un nivel arriba
        extracted_data_dir = os.path.join(parent_directory, "extracted_data")  # Carpeta 'extracted_data'
        os.makedirs(extracted_data_dir, exist_ok=True)

        # Descargar los datos desde Kaggle
        logging.info(f"Downloading data from Kaggle competition: {kaggle_competition}")
        zip_path = os.path.join(parent_directory, f"{kaggle_competition}.zip")  # Guardar ZIP afuera

        # Configurar y autenticar la API de Kaggle
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(kaggle_competition, path=parent_directory)

        # Extraer los archivos del ZIP
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(parent_directory)
        logging.info(f"Data extracted to {parent_directory}")

        # Eliminar el archivo ZIP
        os.remove(zip_path)
        logging.info(f"ZIP file {zip_path} removed after extraction.")

        # Mover solo los archivos deseados a la carpeta 'extracted_data'
        required_files = ["test_set.csv", "train_set.csv"]
        unwanted_files = ["sample_submission.csv"]  

        extracted_files = []
        for file_name in required_files:
            file_path = os.path.join(parent_directory, file_name)
            if os.path.exists(file_path):
                target_path = os.path.join(extracted_data_dir, file_name)
                os.rename(file_path, target_path)  # Mover archivo
                extracted_files.append(file_name)
                logging.info(f"Moved {file_name} to {extracted_data_dir}")
            else:
                logging.warning(f"File {file_name} not found in the extracted data.")

        ### AQUI IRIA EL script para limpiar y

        # Eliminar cualquier archivo no deseado en el directorio raíz
        # Eliminar archivos no deseados
        for file_name in unwanted_files:
            file_path = os.path.join(parent_directory, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Removed unwanted file: {file_name}")
                
        # Responder con la lista de archivos almacenados
        return func.HttpResponse(
            json.dumps({
                "message": "Data extracted and stored successfully.",
                "extracted_files": extracted_files
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return func.HttpResponse(
            "An internal error occurred.",
            status_code=500
        )