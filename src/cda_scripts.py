
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("loaded")

# Preprocessing
import datetime

def filter_exclude_time(df_proc, time_col="ScheduleTime", exclude_time="2021-02-01"):
    """ Exclude time-period before """

    exclude = (df_proc[time_col] < pd.to_datetime(exclude_time)).values
    return df_proc[~exclude].reset_index()

def add_flight_counts(df_proc, time_col="ScheduleTime"):
    """ Adds total count for flight number that month """

    S = df_proc["ScheduleTime"].dt
    df_proc["FlightCount"] = np.nan
    df_proc["SectorCount"] = np.nan
    #df_proc["MonthCapacity"] = np.nan

    for year in [2021, 2022]:
        for month in range(1, 12+1):

            m = np.logical_and(S.month == month, S.year == year)
            month_data = df_proc[m].copy()
            month_idxs = month_data.index
            
            #month_capacity = month_data["SeatCapacity"].mean()
            #df_proc.loc[month_idxs, "MonthCapacity"] = month_capacity
            
            # Map monthly flights by unique flight number
            data = month_data["FlightNumber"].value_counts()
            d = {m:c for m, c in zip(data.index, data.values)}
            mapped_counts = month_data.loc[month_idxs, "FlightNumber"].map(d).values
            df_proc.loc[month_idxs, "FlightCount"] = mapped_counts

            # Map monthly sector flights by sector
            data = month_data["Sector"].value_counts()
            d = {m:c for m, c in zip(data.index, data.values)}
            mapped_counts = month_data.loc[month_idxs, "Sector"].map(d).values
            df_proc.loc[month_idxs, "SectorCount"] = mapped_counts
            

    return df_proc

def add_date_features(df_proc, time_col="ScheduleTime"):
    """Adds one-hot encoded time features from datetime column to df_proc"""

    time_series = df_proc[time_col]
    S = time_series.dt
    S_time = S.hour + (S.minute/60)

    df_time = pd.DataFrame({"t_month":S.month,
                            "t_day":S.day,
                            "t_time":S_time,
                            "t_weekofyear":S.isocalendar().week,
                            "t_dayofyear":S.day_of_year,
                            "t_morning":np.logical_and(S_time >= 0, S_time < 12).astype(np.int64),
                            "t_afternoon":np.logical_and(S_time >= 12, S_time < 16).astype(np.int64),
                            "t_evening":np.logical_and(S_time >= 16, S_time < 20).astype(np.int64),
                            "t_night":np.logical_and(S_time >= 20, S_time <= 24).astype(np.int64),
                            "t_weekend":S.weekday.isin([4,5,6]),
                            "t_weekday":S.weekday.isin([4,5,6]),
                            "t_is_month_start":S.is_month_start,
                            "t_is_month_end":S.is_month_end,
                            "t_is_year_start":S.is_year_start,
                            "t_is_year_end":S.is_year_end,
                            "t_is_quarter_start":S.is_quarter_start,
                            "t_is_quarter_end":S.is_quarter_end
                            })

    # Year 2021, 2022
    df_year = pd.get_dummies(pd.Categorical(S.year)).astype(np.int64)
    df_year.columns = ["t_year_" + str(s) for s in list(df_year.columns.values)]

    # Weekday
    df_weekday = pd.get_dummies(pd.Categorical(S.dayofweek)).astype(np.int64)
    df_weekday.columns = ["t_dayofweek_" + str(s+1) for s in list(df_weekday.columns.values)]

    df_out = pd.concat([df_proc, df_year, df_weekday, df_time], axis=1)
    df_out.drop(time_col, axis=1)

    return df_out

def add_cat_features(df_proc, cat_cols):
    """Adds one-hot encoded columns from cat_columns to df_proc"""
    
    #df_cat = df_raw_realized[["Airline", "Destination", "AircraftType", "FlightType", "Sector"]]
    #include = ["Destination", "Sector", "FlightType", "AircraftType"]

    dfs = []
    for col in cat_cols:
        df = pd.get_dummies(pd.Categorical(df_proc[col]))
        df.columns = [col + "_" + str(s) for s in list(df.columns.values)]
        dfs.append(df)

    df_onehot = pd.concat(dfs, axis=1)
    df_out = pd.concat([df_proc, df_onehot], axis=1)
    df_out = df_out.drop(cat_cols, axis=1)

    return df_out

def normalize_minmax_cols(df_proc, norm_cols):
    """ Min-max normalizes norm_cols in df_proc """

    # Normalize
    for col in norm_cols:
        x = df_proc[col]
        x_norm = (x - x.min())
        x_norm = x_norm / x_norm.max()

        df_proc[col] = x_norm

    return df_proc

def add_time_delta(df_proc, end_time="2022-03-31", time_col="ScheduleTime"):
    """Adds column representing time proximity to end date in fractional months"""
    end_time = pd.to_datetime(end_time)

    df = df_proc[time_col]
    time_delta =  (df - end_time) / np.timedelta64(1, "M") * -1
    time_delta = 1/time_delta
    time_delta = (time_delta - time_delta.min())

    df_proc["t_delta"] = time_delta
    
    return df_proc

def remove_columns(df_proc, exclude_cols):
    """ Removes columns from df_proc if present """

    exclude_cols = pd.Series(exclude_cols)
    exclude_cols = exclude_cols[exclude_cols.isin(df_proc.columns)].values

    df_proc = df_proc.drop(exclude_cols, axis=1)

    return df_proc


# Machine learning
def create_trainval(df_proc, val_months, val_year="t_year_2021", exclude_cols=["ScheduleTime"], y_col="LoadFactor"):
    """ Creates training and validation datasets by selected months/years """

    # FloatingPointError
    m1 = df_proc[val_year] == True
    m2 = df_proc["t_month"].isin(val_months)
    mc = np.logical_and(m1, m2)

    val_idxs = df_proc[~mc].index
    train_idxs = df_proc[mc].index

    # Remove target y column and any other columns to exclude
    y_col = "LoadFactor"
    exclude_cols = [y_col] + exclude_cols
    print(f"Excluding cols in features: {exclude_cols}")

    dataset_X = df_proc.drop(exclude_cols, axis=1)
    dataset_y = df_proc[y_col]

    X_train, y_train = dataset_X.loc[train_idxs], dataset_y.loc[train_idxs]
    X_val, y_val = dataset_X.loc[val_idxs], dataset_y.loc[val_idxs]

    return X_train, y_train, X_val, y_val

import sklearn.metrics as metrics
import scipy as sc
def test_performance_continuous(y_pred, y_true, text=""):
    # 
    mae = np.round(metrics.mean_absolute_error(y_pred, y_true), 3)

    # Calculate pearson correlation beween predicted values (y_hat) and true values (y_true)
    pearson = sc.stats.pearsonr(y_true, y_pred)
    pearson, p = np.round(pearson[0], 3), pearson[1]

    print(f"{text} MAE: {mae}, Pearson {pearson}")

def get_top_coef_perc(model, features, exclude_cols=[], exclude_zero=True, perc=False):
    """ Returns top model coefficients filtered for strings in exclude_cols """
    try:
        coef = model.coef_
    except:
        coef = model.feature_importances_

    coef = pd.Series(coef, index=features.columns).abs().sort_values()[::-1]
    top_coef = coef.copy()

    if exclude_zero:
        top_coef = coef[coef != 0]

    for s in exclude_cols:
        top_coef = top_coef[~top_coef.index.str.contains(s)]

    # Percentage
    perc = top_coef / np.sum(coef)
    p = np.round(np.sum(perc)*100, 2) 
    print(f"Filtered feature importance {p} %")

    return perc

def get_coef(model, features, sort_abs=False, exclude_zero=False):
    """ Get model coefficients for each feature """

    try:
        coef = model.coef_
    except:
        coef = model.feature_importances_

    coef = pd.Series(coef, index=features.columns)

    if exclude_zero:
        coef = coef[coef != 0]

    if sort_abs:
        coef = coef.sort_values()[::-1]

    return coef

def fit_model(model, X_train, y_train, X_val, y_val, verbose=0):
    """ Fits model """
    # Fit

    model.fit(X_train.values, y_train.values)

    y_pred_train = model.predict(X_train.values)
    test_performance_continuous(y_pred_train, y_train, text="Train:")

    y_pred_val = model.predict(X_val.values)
    test_performance_continuous(y_pred_val, y_val, text="Valid:")

    exclude_cols = ["Aircraft", "Destination", "Airline"]
    exclude_cols = []
    
    try:
        get_top_coef_perc(model, X_train, exclude_cols)
    except Exception as E:
        print("{E}")

    return model

def select_features(X_train, y_train, X_val, y_val, features):
    """ Selects given features in dataset """
    return X_train[features], y_train, X_val[features], y_val


def filter_features(X_train, y_train, X_val, y_val, filter_list):
    """ Filters features if string contains keywords in filter_list """

    features = X_train.columns
    for f in filter_list:
        features = features[~features.str.contains(f)]

    X_train2, y_train2, X_val2, y_val2 = select_features(X_train, y_train, X_val, y_val, features)

    return X_train2, y_train2, X_val2, y_val2

