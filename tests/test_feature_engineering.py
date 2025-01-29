import pytest
import pandas as pd
import numpy as np
from Cleaning.feature_engineering import FeatureEngineering

@pytest.fixture
def feature_eng():
    """Instancia de la clase FeatureEngineering para reutilizar en los tests."""
    return FeatureEngineering()

def test_feature_engineering_basic(feature_eng):
    """Prueba básica para verificar la creación de nuevas características."""
    data = {
        "start_time": pd.to_datetime(["2024-01-01 08:30:00", "2024-01-06 15:45:00"]),
        "end_time": pd.to_datetime(["2024-01-01 08:45:00", "2024-01-06 16:15:00"]),
        "start_lat": [37.7749, 34.0522],
        "start_lon": [-122.4194, -118.2437],
        "end_lat": [37.8044, 34.0522],
        "end_lon": [-122.2711, -118.2437],
    }
    input_df = pd.DataFrame(data)

    output_df = feature_eng.feature_engineering(input_df)

    # Verificar que las columnas generadas existen
    expected_columns = ["trip_duration_calculated", "is_weekend", "distance", "time_slot", "day_of_week", "start_hour"]
    for col in expected_columns:
        assert col in output_df.columns, f"La columna {col} no fue creada."

    # Verificar el cálculo de la duración del viaje (15 min y 30 min)
    assert output_df.loc[0, "trip_duration_calculated"] == 15.0, "El cálculo de trip_duration_calculated es incorrecto."
    assert output_df.loc[1, "trip_duration_calculated"] == 30.0, "El cálculo de trip_duration_calculated es incorrecto."

    # Verificar que is_weekend es True para el sábado
    assert output_df.loc[0, "is_weekend"] == False, "is_weekend debería ser False para un lunes."
    assert output_df.loc[1, "is_weekend"] == True, "is_weekend debería ser True para un sábado."

def test_impute_distance_median(feature_eng):
    """Prueba de imputación de distancia usando la estrategia 'median'."""
    data = {
        "start_station": ["Station A", "Station A", "Station B"],
        "distance": [10.0, np.nan, 20.0],
    }
    input_df = pd.DataFrame(data)

    output_df = feature_eng.impute_distance(input_df)

    # Verificar que el valor nulo se imputa con la mediana global (15.0)
    assert output_df.loc[1, "distance"] == 15.0, "La imputación de distancia con median falló."

def test_impute_distance_missing_column(feature_eng):
    """Prueba cuando falta la columna 'distance'."""
    data = {
        "start_station": ["Station A", "Station B"],
        # Falta la columna 'distance'
    }
    input_df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="La columna 'distance' no está en el DataFrame."):
        feature_eng.impute_distance(input_df)

def test_add_features(feature_eng):
    """Prueba para verificar la adición de nuevas características en add_features."""
    data = {
        "start_station": ["Station A", "Station A", "Station B"],
        "trip_duration_calculated": [10, 20, 30],
        "distance": [1.0, 2.0, 3.0],
    }
    input_df = pd.DataFrame(data)

    output_df = feature_eng.add_features(input_df)

    # Verificar que las nuevas columnas existen
    assert "avg_trip_duration_by_station" in output_df.columns, "No se creó avg_trip_duration_by_station."
    assert "duration_distance_ratio" in output_df.columns, "No se creó duration_distance_ratio."

    # Verificar cálculos
    assert output_df.loc[0, "avg_trip_duration_by_station"] == 15.0, "El promedio de duración por estación es incorrecto."
    assert output_df.loc[2, "avg_trip_duration_by_station"] == 30.0, "El promedio de duración por estación es incorrecto."

    # Verificar cálculo de ratio (viaje / distancia)
    assert output_df.loc[0, "duration_distance_ratio"] == pytest.approx(10.0, rel=1e-5), "El cálculo de duration_distance_ratio es incorrecto."
    assert output_df.loc[1, "duration_distance_ratio"] == pytest.approx(10.0, rel=1e-5), "El cálculo de duration_distance_ratio es incorrecto."
