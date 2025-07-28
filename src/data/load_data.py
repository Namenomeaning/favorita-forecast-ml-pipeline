import pandas as pd
import numpy as np
from pathlib import Path

def load_raw_data():
    """
    Load all raw data files from data/csv directory.
    
    Returns:
        tuple: (train, test, stores, transactions, oil, holidays) dataframes
    """
    raw_path = Path(__file__).parent.parent.parent / "data" / "csv"
    
    # Load main datasets
    train = pd.read_csv(raw_path / "train.csv")
    test = pd.read_csv(raw_path / "test.csv")
    stores = pd.read_csv(raw_path / "stores.csv")
    transactions = pd.read_csv(raw_path / "transactions.csv")
    oil = pd.read_csv(raw_path / "oil.csv")
    holidays = pd.read_csv(raw_path / "holidays_events.csv")
    
    # Convert date columns
    for df in [train, test, transactions, oil, holidays]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
    
    # Optimize data types
    train['onpromotion'] = train['onpromotion'].astype("float16")
    train['sales'] = train['sales'].astype("float32")
    stores['cluster'] = stores['cluster'].astype("int8")
    
    return train, test, stores, transactions, oil, holidays