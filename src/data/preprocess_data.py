import pandas as pd
import numpy as np

def preprocess_oil_data(oil):
    """
    Handle missing oil prices with interpolation.
    
    Args:
        oil (pd.DataFrame): Oil prices dataframe
        
    Returns:
        pd.DataFrame: Preprocessed oil data with interpolated missing values
    """
    oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    oil["dcoilwtico"] = oil.dcoilwtico.interpolate()
    return oil

def preprocess_holidays(holidays, stores):
    """
    Process holidays and events data by handling transferred holidays and splitting by scope.
    
    Args:
        holidays (pd.DataFrame): Holidays and events dataframe
        stores (pd.DataFrame): Stores dataframe (for reference)
        
    Returns:
        tuple: (events, national, regional, local) dataframes
    """
    # Handle transferred holidays
    tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis=1)
    tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis=1)
    
    if not tr1.empty and not tr2.empty:
        tr = pd.concat([tr1, tr2], axis=1)
        tr = tr.iloc[:, [5,1,2,3,4]]
        holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis=1)
        holidays = pd.concat([holidays, tr])
    
    # Simplify holiday types
    holidays["type"] = holidays["type"].replace(["Additional", "Bridge"], "Holiday")
    
    # Split by scope
    events = holidays[holidays.type == "Event"][["date", "description"]].rename(columns={"description": "events"})
    national = holidays[holidays.locale == "National"][["date", "description"]].rename(columns={"description": "holiday_national"})
    regional = holidays[holidays.locale == "Regional"][["date", "locale_name", "description"]].rename(columns={"locale_name": "state", "description": "holiday_regional"})
    local = holidays[holidays.locale == "Local"][["date", "locale_name", "description"]].rename(columns={"locale_name": "city", "description": "holiday_local"})
    
    return events, national, regional, local
