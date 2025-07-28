import pandas as pd
from pathlib import Path

def save_processed_data(train_processed, test_processed, zero_predictions):
    """
    Save processed data to featured directory.
    
    Args:
        train_processed (pd.DataFrame): Processed training data
        test_processed (pd.DataFrame): Processed test data
        zero_predictions (pd.DataFrame): Zero predictions data
    """
    # Use absolute path for featured directory
    featured_dir = Path(__file__).parent.parent.parent / "data" / "featured"
    featured_dir.mkdir(parents=True, exist_ok=True)

    print("Saving processed data...")
    train_processed.to_parquet(featured_dir / "train_featured.parquet", index=False)
    test_processed.to_parquet(featured_dir / "test_featured.parquet", index=False)

    if not zero_predictions.empty:
        zero_predictions.to_parquet(featured_dir / "zero_predictions.parquet", index=False)

    print(f"Train shape: {train_processed.shape}")
    print(f"Test shape: {test_processed.shape}")
    print(f"Zero predictions: {len(zero_predictions) if not zero_predictions.empty else 0}")
