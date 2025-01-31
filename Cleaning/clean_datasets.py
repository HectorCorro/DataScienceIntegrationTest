import pandas as pd
import numpy as np

class DataClener:
    """
    Clase para limpiar datos, incluyendo el manejo de valores nulos en columnas geográficas.
    """

    def __init__(self):
        pass

    def cleaning_data(self, df):
        """
        Limpia los datos reemplazando valores nulos en coordenadas geográficas
        con la mediana del grupo correspondiente o con 0 si no hay valores disponibles.
        
        Args:
            df (pd.DataFrame): DataFrame con datos a limpiar.
        
        Returns:
            pd.DataFrame: DataFrame con valores limpios en columnas geográficas.
        """
        geo_columns = ['start_lat', 'start_lon', 'end_lat', 'end_lon']

        for col in geo_columns:
            if 'start' in col:
                df[col] = df.groupby('start_station')[col].transform(
                    lambda x: x.fillna(x.median()) if not x.median() is np.nan else x.fillna(0)
                    )
            else:
                df[col] = df.groupby('end_station')[col].transform(
                    lambda x: x.fillna(x.median()) if not x.median() is np.nan else x.fillna(0)
                    )

        for col in geo_columns:
            if df[col].isnull().all():
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        return df