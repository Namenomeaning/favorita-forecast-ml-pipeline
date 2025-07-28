import pandas as pd


def merge_external_data(df, stores, transactions, oil, events, national, regional, local):
    """
    Merge all external data sources with main dataframe.

    Args:
        df (pd.DataFrame): Main dataframe
        stores (pd.DataFrame): Stores metadata
        transactions (pd.DataFrame): Transaction data
        oil (pd.DataFrame): Oil prices data
        events (pd.DataFrame): Events data
        national (pd.DataFrame): National holidays data
        regional (pd.DataFrame): Regional holidays data
        local (pd.DataFrame): Local holidays data

    Returns:
        pd.DataFrame: Merged dataframe with all features
    """
    # Merge store info
    df = pd.merge(df, stores, on="store_nbr", how="left")

    # Merge transactions
    df = pd.merge(df, transactions, on=["date", "store_nbr"], how="left")

    # Merge oil prices
    df = pd.merge(df, oil, on="date", how="left")

    # Merge holidays and events
    df = pd.merge(df, national, on="date", how="left")
    df = pd.merge(df, regional, on=["date", "state"], how="left")
    df = pd.merge(df, local, on=["date", "city"], how="left")

    # Create holiday binary features
    df["holiday_national_binary"] = df.holiday_national.notnull().astype("int8")
    df["holiday_regional_binary"] = df.holiday_regional.notnull().astype("int8")
    df["holiday_local_binary"] = df.holiday_local.notnull().astype("int8")

    # Create workday feature
    df["workday"] = (~((df.holiday_national_binary == 1) |
                       (df.holiday_regional_binary == 1) |
                       (df.holiday_local_binary == 1) |
                       (df.is_weekend == 1))).astype("int8")

    # Event features (simplified)
    df = pd.merge(df, events, on="date", how="left")
    df["has_event"] = df.events.notnull().astype("int8")

    return df
