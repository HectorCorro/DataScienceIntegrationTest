import pandas as pd
import numpy as np

class DataClener:

    def _init_(self):
        pass
 
    def cleaning_data(self, df):

        def parse_dates(date):
            try:
                return pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    return pd.to_datetime(date, format='%m/%d/%Y %H:%M')
                except ValueError:
                    return pd.NaT
        
        df['start_time'] = df['start_time'].apply(parse_dates)
        df['end_time'] = df['end_time'].apply(parse_dates)

        geo_columns = ['start_lat', 'start_lon', 'end_lat', 'end_lon']

        for col in geo_columns:
            if 'start' in col:
                df[col] = df.groupby('start_station')[col].transform(lambda x: x.fillna(x.median()))
                df[col] = df.groupby('start_station')[col].transform(lambda x: x.fillna(x.median()))
            else:
                df[col] = df.groupby('end_station')[col].transform(lambda x: x.fillna(x.median()))
                df[col] = df.groupby('end_station')[col].transform(lambda x: x.fillna(x.median()))

        for col in geo_columns:
            df[col].fillna(df[col].median(), inplace=True)
            df[col].fillna(df[col].median(), inplace=True)

        return df