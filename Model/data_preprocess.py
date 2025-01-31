import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataPreprocess:
    """
    Clase para preprocesar datos, incluyendo codificación de variables categóricas
    y manejo de valores atípicos.
    """

    def __init__(self):
        pass
    
    def pre_process_df(self, df, Train=False):
        """
        Preprocesa un DataFrame codificando variables categóricas.
        
        Args:
            df (pd.DataFrame): DataFrame con datos a procesar.
            Train (bool): Indica si se está procesando el conjunto de entrenamiento para ajuste de etiquetas.
        
        Returns:
            tuple: DataFrame procesado y lista de nombres de columnas codificadas.
        """
        le = LabelEncoder()
        encoder = OneHotEncoder()
        if Train:            
            df['passholder_type_encoded'] = le.fit_transform(df['passholder_type'])
           
        encoded_days = encoder.fit_transform(df[['day_of_week', 'time_slot']]).toarray()
        df['trip_route_category_encoded'] = le.fit_transform(df['trip_route_category'])
        encoded_columns = encoder.get_feature_names_out(['day_of_week', 'time_slot'])
        df = pd.concat([df, pd.DataFrame(encoded_days, columns=encoded_columns, index=df.index)], axis=1)
        return df, encoded_columns
    
    def handle_outliers(self, df):
        """
        Filtra valores atípicos en las columnas 'trip_duration_calculated' y 'distance'
        utilizando el rango intercuartil (IQR).
        
        Args:
            df (pd.DataFrame): DataFrame con datos a procesar.
        
        Returns:
            pd.DataFrame: DataFrame sin valores atípicos.
        """
        q1_duration = df['trip_duration_calculated'].quantile(0.25)
        q3_duration = df['trip_duration_calculated'].quantile(0.75)
        iqr_duration = q3_duration - q1_duration
        lower_bound_duration = q1_duration - 2 * iqr_duration
        upper_bound_duration = q3_duration + 2 * iqr_duration
        
        q1_distance = df['distance'].quantile(0.25)
        q3_distance = df['distance'].quantile(0.75)
        iqr_distance = q3_distance - q1_distance
        lower_bound_distance = q1_distance - 2 * iqr_distance
        upper_bound_distance = q3_distance + 2 * iqr_distance

        df = df[
            (df['trip_duration_calculated'] >= lower_bound_duration) & 
            (df['trip_duration_calculated'] <= upper_bound_duration) & 
            (df['distance'] >= lower_bound_distance) & 
            (df['distance'] <= upper_bound_distance)
        ]
        return df
    
