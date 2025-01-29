# utils.py
import pandas as pd

def parse_dates(date):
    if pd.isnull(date):  # Manejar valores nulos
        return pd.NaT
    try:
        return pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S', errors='raise')
    except ValueError:
        try:
            return pd.to_datetime(date, format='%m/%d/%Y %H:%M', errors='raise')
        except ValueError:
            return pd.NaT