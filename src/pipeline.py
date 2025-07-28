import pandas as pd
import numpy as np
from data.load_data import load_raw_data
from data.preprocess_data import preprocess_oil_data, preprocess_holidays
from data.clean_data import remove_inactive_stores, handle_zero_sales
from data.feature_engineering import create_time_features, create_lag_features
from data.merge_data import merge_external_data
from data.save_data import save_processed_data

def main():
    """
    Main pipeline function to process all data from csv to featured format.
    """
    print("Starting data preprocessing pipeline...")
    
    # Load raw data
    print("Loading raw data...")
    train, test, stores, transactions, oil, holidays = load_raw_data()
    
    # Preprocess external data
    print("Preprocessing external data...")
    oil = preprocess_oil_data(oil)
    events, national, regional, local = preprocess_holidays(holidays, stores)
    
    # Clean training data
    print("Cleaning training data...")
    train = remove_inactive_stores(train)
    train, zero_predictions = handle_zero_sales(train, test)
    
    # Combine train and test for consistent processing
    test["sales"] = np.nan
    test["sales"] = test["sales"].astype("float32")
    combined = pd.concat([train, test], ignore_index=True)
    
    # Create features
    print("Creating features...")
    combined = create_time_features(combined)
    combined = merge_external_data(combined, stores, transactions, oil, events, national, regional, local)
    combined = create_lag_features(combined)
    
    # Create item_id and rename date
    combined["item_id"] = combined["store_nbr"].astype(str) + "_" + combined["family"]
    combined = combined.rename(columns={"date": "timestamp"})
    
    # Split back into train and test
    train_processed = combined[combined.sales.notnull()].copy()
    test_processed = combined[combined.sales.isnull()].drop("sales", axis=1).copy()
    
    # Update zero predictions format
    if not zero_predictions.empty:
        zero_predictions["item_id"] = zero_predictions["store_nbr"].astype(str) + "_" + zero_predictions["family"]
        zero_predictions = zero_predictions.rename(columns={"date": "timestamp"})
    
    # Save processed data
    save_processed_data(train_processed, test_processed, zero_predictions)
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()