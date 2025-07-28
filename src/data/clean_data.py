import pandas as pd

def remove_inactive_stores(train):
    """
    Remove periods before stores opened based on analysis.
    
    Args:
        train (pd.DataFrame): Training dataframe
        
    Returns:
        pd.DataFrame: Cleaned training data with inactive periods removed
    """
    initial_shape = train.shape[0]
    
    # Store opening dates based on notebook analysis
    store_openings = {
        52: "2017-04-20", 22: "2015-10-09", 42: "2015-08-21",
        21: "2015-07-24", 29: "2015-03-20", 20: "2015-02-13",
        53: "2014-05-29", 36: "2013-05-09"
    }
    
    for store_nbr, open_date in store_openings.items():
        train = train[~((train.store_nbr == store_nbr) & (train.date < open_date))]
    
    print(f"Removed {initial_shape - train.shape[0]} rows from inactive store periods")
    return train

def handle_zero_sales(train, test):
    """
    Identify and separate zero-sales combinations.
    
    Args:
        train (pd.DataFrame): Training dataframe
        test (pd.DataFrame): Test dataframe
        
    Returns:
        tuple: (cleaned_train, zero_predictions) dataframes
    """
    zero_combos = train.groupby(["store_nbr", "family"]).sales.sum().reset_index()
    zero_combos = zero_combos[zero_combos.sales == 0][["store_nbr", "family"]]
    
    # Remove zero combinations from training data
    train = train.merge(zero_combos, how="left", indicator=True)
    train = train[train._merge == "left_only"].drop("_merge", axis=1)
    
    # Create zero predictions for test period
    zero_predictions = []
    for _, row in zero_combos.iterrows():
        zero_pred = pd.DataFrame({
            "date": pd.date_range("2017-08-16", "2017-08-31"),
            "store_nbr": row.store_nbr,
            "family": row.family,
            "sales": 0
        })
        zero_predictions.append(zero_pred)
    
    if zero_predictions:
        zero_predictions = pd.concat(zero_predictions)
    else:
        zero_predictions = pd.DataFrame()
    
    return train, zero_predictions