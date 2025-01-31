import pandas as pd
import numpy as np

class FeatureEngineering:

    def __init__(self, strategy='median'):
        self.strategy = strategy
    
    def feature_engineering(self, df):
        """
        Crea variables de tiempo como duración del viaje, fines de semana, variables de distancia y binning horario.
        :param df: DataFrame a procesar.
        :return: DataFrame con variables extra generadas para modelar fenómenos de viaje.
        """
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
            df['distance'] = df['distance'].fillna(df['distance'].mean())
        elif self.strategy == 'median':
            df['distance'] = df['distance'].fillna(df['distance'].median())
        elif self.strategy == 'grouped_median':
            if 'start_station' not in df.columns:
                raise ValueError("La columna 'start_station' no está en el DataFrame.")
            # Calcular las medianas por grupo
            grouped_medians = df.groupby('start_station')['distance'].median()
            print("Medianas por grupo:")
            print(grouped_medians)

            # Aplicar la mediana del grupo si existe
            df['distance'] = df.groupby('start_station')['distance'].transform(
                lambda x: x.fillna(x.median()) if not pd.isna(x.median()) else x
            )

            # Verificar si aún quedan valores NaN y aplicar la mediana global solo a esos valores
            global_median = df['distance'].median()
            print(f"Mediana global: {global_median}")

            df['distance'] = df['distance'].fillna(global_median)
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")
        
        return df
    
    def add_features(self, df):
        """
        Este método permite escribir la lógica de más features o variables que puedan hacer sentido al negocio.
        :param: Dataframe a procesar.
        :return: DataFrame con variables extra'.
        """
        # Promedio de duración de viajes por estación de inicio
        df['avg_trip_duration_by_station'] = df.groupby('start_station')['trip_duration_calculated'].transform('mean')
        # Interacciones entre variables
        df['duration_distance_ratio'] = df['trip_duration_calculated'] / (df['distance'] + 1e-5)  # Evitar división por cero
        # Se pueden agregar mas variables
        return df