import pytest
import pandas as pd
import numpy as np
from Model.data_preprocess import DataPreprocess

@pytest.fixture
def preprocessor():
    """Instancia de la clase DataPreprocess para reutilizar en los tests."""
    return DataPreprocess()

def test_pre_process_df_basic(preprocessor):
    """Prueba básica para verificar la codificación de variables categóricas."""
    data = {
        "passholder_type": ["Subscriber", "Customer", "Subscriber"],
        "trip_route_category": ["One Way", "Round Trip", "One Way"],
        "day_of_week": ["Monday", "Tuesday", "Wednesday"],
        "time_slot": ["Morning", "Afternoon", "Evening"]
    }
    input_df = pd.DataFrame(data)

    output_df, encoded_columns = preprocessor.pre_process_df(input_df, Train=True)

    # Verificar que las nuevas columnas codificadas existen
    assert "passholder_type_encoded" in output_df.columns, "La columna passholder_type_encoded no fue creada."
    assert "trip_route_category_encoded" in output_df.columns, "La columna trip_route_category_encoded no fue creada."

    # Verificar que las columnas One-Hot Encoding existen
    for col in encoded_columns:
        assert col in output_df.columns, f"La columna {col} no fue creada."

    # Verificar que las columnas originales siguen presentes
    for col in data.keys():
        assert col in output_df.columns, f"La columna original {col} desapareció."

def test_pre_process_df_missing_column(preprocessor):
    """Prueba para verificar el comportamiento si falta passholder_type."""
    data = {
        "trip_route_category": ["One Way", "Round Trip", "One Way"],
        "day_of_week": ["Monday", "Tuesday", "Wednesday"],
        "time_slot": ["Morning", "Afternoon", "Evening"]
    }
    input_df = pd.DataFrame(data)

    # La función debería fallar al intentar codificar passholder_type
    with pytest.raises(KeyError):
        preprocessor.pre_process_df(input_df, Train=True)

def test_handle_outliers(preprocessor):
    """Prueba para verificar la eliminación de valores atípicos con IQR."""
    data = {
        "trip_duration_calculated": [100, 200, 300, 5000],  # 5000 es un outlier
        "distance": [1.5, 2.0, 2.5, 50.0]  # 50.0 es un outlier
    }
    input_df = pd.DataFrame(data)

    output_df = preprocessor.handle_outliers(input_df)

    # Verificar que los outliers específicos fueron eliminados
    assert 5000 not in output_df["trip_duration_calculated"].values, "El outlier en trip_duration_calculated no fue eliminado."
    assert 50.0 not in output_df["distance"].values, "El outlier en distance no fue eliminado."

    # Verificar que los valores dentro del IQR se mantuvieron
    assert len(output_df) == 3, f"Se eliminaron valores dentro del rango permitido. Filas restantes: {len(output_df)}"

def test_handle_outliers_no_outliers(preprocessor):
    """Prueba cuando no hay outliers en el DataFrame."""
    data = {
        "trip_duration_calculated": [100, 150, 200, 250, 300],
        "distance": [1.0, 1.5, 2.0, 2.5, 3.0]
    }
    input_df = pd.DataFrame(data)

    output_df = preprocessor.handle_outliers(input_df)

    # Verificar que el número de filas sigue igual
    assert len(output_df) == len(input_df), "Se eliminaron datos incorrectamente cuando no había outliers."
