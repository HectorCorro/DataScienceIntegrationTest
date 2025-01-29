import pandas as pd
import numpy as np

class DataClener:

    def __init__(self):
        pass

    def cleaning_data(self, df):

        #df['start_time'] = df['start_time'].apply(parse_dates)
        #df['end_time'] = df['end_time'].apply(parse_dates)

        geo_columns = ['start_lat', 'start_lon', 'end_lat', 'end_lon']

        for col in geo_columns:
            if 'start' in col:
                df[col] = df.groupby('start_station')[col].transform(lambda x: x.fillna(x.median()))
            else:
                df[col] = df.groupby('end_station')[col].transform(lambda x: x.fillna(x.median()))

        for col in geo_columns:
            df[col].fillna(df[col].median(), inplace=True)

        return df