import pandas as pd
import numpy as np

class FeatureEngineering:

    def __init__(self, strategy='median'):
        self.strategy = strategy
    
    def feature_engineering(self, df):

        df['trip_duration_calculated'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
        df['is_weekend'] = df['start_time'].dt.weekday >= 5
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0  # Radio de la Tierra en kilómetros
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c
        
        df['distance'] = haversine(
            df['start_lat'], df['start_lon'], df['end_lat'], df['end_lon']
        )

        df['start_hour'] = df['start_time'].dt.hour
        bins = [0, 6, 12, 18, 24]
        labels = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
        df['time_slot'] = pd.cut(df['start_hour'], bins=bins, labels=labels, right=False)

        df['day_of_week'] = df['start_time'].dt.day_name()

        return df
    
    def impute_distance(self, df):
        """
        Imputa valores faltantes en la columna 'distance'.
        :param df: DataFrame a procesar.
        :return: DataFrame con valores imputados en 'distance'.
        """
        if 'distance' not in df.columns:
            raise ValueError("La columna 'distance' no está en el DataFrame.")

        if self.strategy == 'mean':
            df['distance'].fillna(df['distance'].mean(), inplace=True)
        elif self.strategy == 'median':
            df['distance'].fillna(df['distance'].median(), inplace=True)
        elif self.strategy == 'grouped_median':
            if 'start_station' not in df.columns:
                raise ValueError("La columna 'start_station' no está en el DataFrame.")
            # Imputar usando la mediana agrupada por estación
            df['distance'] = df.groupby('start_station')['distance'].transform(
                lambda x: x.fillna(x.median())
            )
            # Rellenar valores restantes con la mediana global
            df['distance'].fillna(df['distance'].median(), inplace=True)
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")
        
        return df
    
    def add_features(self, df):
        # Promedio de duración de viajes por estación de inicio
        df['avg_trip_duration_by_station'] = df.groupby('start_station')['trip_duration_calculated'].transform('mean')
        # Interacciones entre variables
        df['duration_distance_ratio'] = df['trip_duration_calculated'] / (df['distance'] + 1e-5)  # Evitar división por cero
        # Se pueden agregar mas variables
        return df