import pytest
import pandas as pd
import numpy as np
from Cleaning.clean_datasets import DataClener

@pytest.fixture
def cleaner():
    """Instancia de la clase DataClener para reutilizar en los tests."""
    return DataClener()

def test_cleaning_data_basic(cleaner):
    """Prueba básica: Verificar imputación correcta de valores nulos."""
    data = {
        "start_station": ["Station A", "Station A", "Station B", "Station B", "Station C"],
        "end_station": ["Station X", "Station X", "Station Y", "Station Y", "Station Z"],
        "start_lat": [np.nan, 40.7128, np.nan, 37.7749, 34.0522],
        "start_lon": [-74.0060, np.nan, -122.4194, np.nan, np.nan],
        "end_lat": [np.nan, np.nan, 37.7749, np.nan, 34.0522],
        "end_lon": [-74.0060, np.nan, -122.4194, -122.4194, np.nan],
    }
    input_df = pd.DataFrame(data)
    output_df = cleaner.cleaning_data(input_df)

    geo_columns = ["start_lat", "start_lon", "end_lat", "end_lon"]

    # Verificar que no hay valores nulos
    for col in geo_columns:
        assert output_df[col].isnull().sum() == 0, f"{col} contiene valores nulos."

    # Verificar imputaciones basadas en medianas de grupo
    station_a_start_lat_median = input_df[input_df["start_station"] == "Station A"]["start_lat"].median()
    assert output_df.loc[0, "start_lat"] == station_a_start_lat_median, "La imputación de start_lat falló para Station A."

def test_cleaning_data_empty_df(cleaner):
    """Prueba con un DataFrame vacío."""
    input_df = pd.DataFrame(columns=["start_station", "end_station", "start_lat", "start_lon", "end_lat", "end_lon"])
    output_df = cleaner.cleaning_data(input_df)

    # El DataFrame de salida debe ser igual al de entrada (sin errores)
    assert output_df.equals(input_df), "El DataFrame vacío no se manejó correctamente."

def test_cleaning_data_all_nulls(cleaner):
    """Prueba con todas las columnas geográficas como valores nulos."""
    data = {
        "start_station": ["Station A", "Station B", "Station C"],
        "end_station": ["Station X", "Station Y", "Station Z"],
        "start_lat": [np.nan, np.nan, np.nan],
        "start_lon": [np.nan, np.nan, np.nan],
        "end_lat": [np.nan, np.nan, np.nan],
        "end_lon": [np.nan, np.nan, np.nan],
    }
    input_df = pd.DataFrame(data)
    output_df = cleaner.cleaning_data(input_df)

    # Verificar que no hay valores nulos después de la imputación
    geo_columns = ["start_lat", "start_lon", "end_lat", "end_lon"]
    for col in geo_columns:
        assert output_df[col].isnull().sum() == 0, f"{col} contiene valores nulos."

    # Verificar que se usaron las medianas globales
    for col in geo_columns:
        global_median = input_df[col].median()
        assert output_df[col].median() == global_median, f"La imputación global de {col} no es correcta."

def test_cleaning_data_missing_column(cleaner):
    """Prueba cuando faltan columnas requeridas en el DataFrame."""
    data = {
        "start_station": ["Station A", "Station B", "Station C"],
        "end_station": ["Station X", "Station Y", "Station Z"],
        "start_lat": [np.nan, 40.7128, 34.0522],
        # Falta 'start_lon'
        "end_lat": [37.7749, 37.7749, 34.0522],
        "end_lon": [-122.4194, -122.4194, -118.2437],
    }
    input_df = pd.DataFrame(data)

    # Verificar que se lanza un error KeyError
    with pytest.raises(KeyError):
        cleaner.cleaning_data(input_df)