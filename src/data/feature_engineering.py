import pandas as pd
import numpy as np


def create_time_features(df):
    """
    Create time-based features from date column.

    Args:
        df (pd.DataFrame): Dataframe with date column

    Returns:
        pd.DataFrame: Dataframe with additional time features
    """
    df = df.copy()
    df['month'] = df.date.dt.month.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
    df['week_of_year'] = df.date.dt.isocalendar().week.astype("int8")
    df['day_of_week'] = (df.date.dt.dayofweek + 1).astype("int8")
    df['year'] = df.date.dt.year.astype("int32")
    df["is_weekend"] = (df.date.dt.weekday >= 5).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")

    # Wage day feature (15th and last day of month)
    df["wage_day"] = ((df['is_month_end'] == 1) | (
        df["day_of_month"] == 15)).astype("int8")

    return df


def create_lag_features(df):
    """
    Create simple lag features for sales.

    Args:
        df (pd.DataFrame): Dataframe with sales data

    Returns:
        pd.DataFrame: Dataframe with lag features
    """
    df = df.sort_values(["store_nbr", "family", "date"])

    # Basic lag features (safe for 15-day forecast)
    for lag in [16, 30, 365]:
        df[f"sales_lag_{lag}"] = df.groupby(["store_nbr", "family"])[
            "sales"].shift(lag)

    # Simple moving averages
    for window in [7, 14, 30]:
        df[f"sales_ma_{window}_lag16"] = (df.groupby(["store_nbr", "family"])["sales"]
                                          .rolling(window).mean().shift(16).values)

    return df
