import azure.functions as func
import pandas as pd
import requests
import datetime
import json
import logging
import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import LabelEncoder


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Cleaning.clean_datasets import DataClener
from Cleaning.utils import parse_dates
from Cleaning.feature_engineering import FeatureEngineering
from Model.data_preprocess import DataPreprocess
from Model.model_develop import ModelDevelop

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
        #repo_directory = os.path.dirname(os.path.abspath(__file__))  # Directorio actual
        #parent_directory = os.path.dirname(repo_directory)  # Un nivel arriba
        BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
        extracted_data_dir = os.path.join(BASE_DIR, "extracted_data")  # Carpeta 'extracted_data'
        os.makedirs(extracted_data_dir, exist_ok=True)

        # Descargar los datos desde Kaggle
        logging.info(f"Downloading data from Kaggle competition: {kaggle_competition}")
        zip_path = os.path.join(BASE_DIR, f"{kaggle_competition}.zip")  # Guardar ZIP afuera

        # Configurar y autenticar la API de Kaggle
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(kaggle_competition, path=BASE_DIR)

        # Extraer los archivos del ZIP
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)
        logging.info(f"Data extracted to {BASE_DIR}")

        # Eliminar el archivo ZIP
        os.remove(zip_path)
        logging.info(f"ZIP file {zip_path} removed after extraction.")

        cleaner = DataClener()

        # Mover solo los archivos deseados a la carpeta 'extracted_data'
        required_files = ["test_set.csv", "train_set.csv"]
        unwanted_files = ["sample_submission.csv"]  

        extracted_files = []
        for file_name in required_files:
            file_path = os.path.join(BASE_DIR, file_name)
            if os.path.exists(file_path):

                df = pd.read_csv(file_path)
                df['start_time'] = df['start_time'].apply(parse_dates)
                df['end_time'] = df['end_time'].apply(parse_dates)
                ### modificar para limpieza
                df = cleaner.cleaning_data(df)

                #if file_name == "train_set.csv":
                    #logging.info(f"Applying additional cleaning to {file_name}")
                    #df = df.dropna(subset=['passholder_type'])  # Eliminar filas con valores nulos en 'passholder_type'

                target_path = os.path.join(extracted_data_dir, file_name)
                df.to_csv(target_path, index=False)
                #os.rename(file_path, target_path)  # Mover archivo
                shutil.copy2(file_path, target_path)  # Copia con metadata
                os.remove(file_path)  # Luego eliminar el original
                extracted_files.append(file_name)
                logging.info(f"Moved {file_name} to {extracted_data_dir}")
            else:
                logging.warning(f"File {file_name} not found in the extracted data.")

        ### AQUI IRIA EL script para limpiar y

        # Eliminar cualquier archivo no deseado en el directorio raíz
        # Eliminar archivos no deseados
        for file_name in unwanted_files:
            file_path = os.path.join(BASE_DIR, file_name)
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
    
@app.route(route="MLPipelineFunction", auth_level=func.AuthLevel.ANONYMOUS)
def MLPipelineFunction(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("ML Pipeline function triggered.")

    try:
        # Definir directorio donde están los datos procesados
        BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
        #repo_directory = os.path.dirname(os.path.abspath(__file__))
        #parent_directory = os.path.dirname(BASE_DIR)  # Un nivel arriba
        extracted_data_dir = os.path.join(BASE_DIR, "extracted_data")

        # Verificar que los archivos existan
        train_file_path = os.path.join(extracted_data_dir, "train_set.csv")
        test_file_path = os.path.join(extracted_data_dir, "test_set.csv")
        if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
            return func.HttpResponse(
                f"Processed files not found in {extracted_data_dir}.",
                status_code=404
            )

        # Leer datos procesados
        logging.info(f"Loading processed datasets from {extracted_data_dir}...")
        data_train = pd.read_csv(train_file_path)
        data_test = pd.read_csv(test_file_path)

        datasets = [data_train, data_test]
        columns = ['start_time', 'end_time']

        # Aplicar parse_dates a cada columna de cada DataFrame
        for df in datasets:
            for col in columns:
                df[col] = df[col].apply(parse_dates)

        data_train = data_train.dropna(subset=['passholder_type'])

        print(data_train.dtypes)
        print(data_test.dtypes)


        # Ingeniería de características
        logging.info("Applying feature engineering...")
        feature_eng = FeatureEngineering(strategy='grouped_median')
        #data_train_cleaned_pre = feature_eng.feature_engineering(data_train)
        #data_test_cleaned_pre = feature_eng.feature_engineering(data_test)
        #data_train_cleaned_pre = feature_eng.impute_distance(data_train)
        #data_test_cleaned_pre = feature_eng.impute_distance(data_test)

        datasets = {
            "data_train": data_train,
            "data_test": data_test,
        }

        cleaned_datasets = {}

        for name, df in datasets.items():
            # Aplicar feature engineering
            logging.info(f"Applying feature engineering to {name}...")
            df_cleaned = feature_eng.feature_engineering(df)
            
            # Imputar distancia
            logging.info(f"Imputing distance for {name}...")
            df_cleaned = feature_eng.impute_distance(df_cleaned)
            
            # Guardar resultados en un diccionario
            cleaned_datasets[name] = df_cleaned

        # Acceder a los datasets limpios
        data_train_cleaned_pre = cleaned_datasets["data_train"]
        data_test_cleaned_pre = cleaned_datasets["data_test"]
        
        output_dir_data = os.path.join(BASE_DIR, "engineered_data")
        os.makedirs(output_dir_data, exist_ok=True)

        # Guardar los datasets procesados
        for name, df in cleaned_datasets.items():
            file_path = os.path.join(output_dir_data, f"{name}_engineered.csv")
            df.to_csv(file_path, index=False)
            logging.info(f"Saved {name} to {file_path}")

        # Preprocesamiento
        logging.info("Preprocessing data...")
        pre_process = DataPreprocess()
        data_train_cleaned = pre_process.handle_outliers(data_train_cleaned_pre)
        data_train_cleaned, encoded_columns = pre_process.pre_process_df(data_train_cleaned_pre, Train=True)
        data_test_cleaned = pre_process.pre_process_df(data_test_cleaned_pre)[0]

        # Definir características
        features = ['start_station', 'trip_duration_calculated', 'start_hour', 'distance', 'is_weekend'] + list(encoded_columns)
        X_train = data_train_cleaned[features]
        print(X_train.isna().sum())
        y_train = data_train_cleaned['passholder_type_encoded']
        X_test = data_test_cleaned[features]

        # División de datos y entrenamiento de modelos
        logging.info("Training models...")
        model_develop = ModelDevelop()
        X_train_internal, X_val, y_train_internal, y_val = model_develop.train_test_split_df(X_train, y_train)
        X_train_balanced, y_train_balanced = model_develop.balance_data(X_train_internal, y_train_internal)
        X_train_balanced = feature_eng.add_features(X_train_balanced)
        X_val = feature_eng.add_features(X_val)
        X_test = feature_eng.add_features(X_test)
        all_features = X_train_balanced.columns.to_list()

        # Entrenamiento y evaluación
        le = LabelEncoder()
        le.fit(data_train_cleaned['passholder_type'])
        rf_model = model_develop.train_and_evaluate_models(X_train_balanced, y_train_balanced, X_val, y_val, le)

        # Predicciones en el conjunto de prueba
        logging.info("Making predictions...")
        y_test_rf = model_develop.predict_on_test_set(rf_model, X_test)
        y_test_rf_labels = le.inverse_transform(y_test_rf)
        logging.info("Saving predictions into csv...")
        output_dir = os.path.join(BASE_DIR, "ml_results")
        os.makedirs(output_dir, exist_ok=True)
        predictions_df = pd.DataFrame({
            'trip_id': data_test_cleaned['trip_id'],
            'predicted_rf': y_test_rf,
            'predicted_label': y_test_rf_labels
        })
        # Guardar predicciones
        predictions_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)

        # Predicción sobre el conjunto de entrenamiento
        y_train_pred = model_develop.predict_on_test_set(rf_model, X_train_balanced)

        # Convertir las predicciones a etiquetas originales
        y_train_pred_labels = le.inverse_transform(y_train_pred)
        y_train_real_labels = le.inverse_transform(y_train_balanced)

        train_predictions_df = pd.DataFrame({
            'y_train_real': y_train_balanced,
            'y_train_real_label': y_train_real_labels,
            'y_train_pred': y_train_pred,
            'y_train_pred_label': y_train_pred_labels
        })

        # Guardar en CSV
        train_predictions_path = os.path.join(output_dir, "train_predictions.csv")
        train_predictions_df.to_csv(train_predictions_path, index=False)

        # Guardar métricas del modelo
        logging.info("Saving metrics...")
        metrics = model_develop.get_model_metrics(rf_model, X_val, y_val, le)
        metrics_path = os.path.join(output_dir, "model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Guardar modelos
        logging.info("Savinf .pkl model file...")
        rf_model_path = os.path.join(output_dir, "rf_model.pkl")
        model_develop.save_model(rf_model, rf_model_path)

        logging.info("Getting & saving feature importances...")

        feature_importance = pd.DataFrame({
            'feature': all_features,
            'importance': rf_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        feature_importance_path = os.path.join(output_dir, "feature_importance.csv")
        feature_importance.to_csv(feature_importance_path, index=False)

        # Responder con éxito
        return func.HttpResponse(
            json.dumps({
                "message": "ML pipeline executed successfully.",
                "predictions_path": predictions_path,
                "rf_model_path": rf_model_path
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