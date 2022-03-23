
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("loaded")

# Preprocessing
import datetime

def filter_exclude_months(df_proc, year, month_list, time_col="ScheduleTime"):
    """ Removing specific months of year from df_proc """
    time = df_proc["ScheduleTime"].dt

    for month in month_list:
        print(year, month)
        exclude = np.logical_and(time.year == year, time.month == month)
        df_proc = df_proc[~exclude]
        print(np.sum(exclude))

    return df_proc

def filter_time_period(df_proc, start_date="2021-06-01", time_col="ScheduleTime"):
    m = df_proc[time_col] >= start_date
    return df_proc[m]


def filter_exclude_time(df_proc, time_col="ScheduleTime", exclude_time="2021-02-01"):
    """ Exclude time-period before """

    exclude = (df_proc[time_col] < pd.to_datetime(exclude_time)).values
    return df_proc[~exclude].reset_index()

def add_flight_counts(df_proc, time_col="ScheduleTime"):
    """ Adds total count for flight number that week """

    def get_mapped_counts(subset_data, col_name):
        """ Maps counts for data """
        data = subset_data[col_name].value_counts()
        d = {m:c for m, c in zip(data.index, data.values)}
        mapped_counts = subset_data[col_name].map(d).values
                
        return mapped_counts

    time = df_proc["ScheduleTime"].dt

    df_proc["FlightCount_week"] = np.nan

    for year in [2021, 2022]:
        for week in sorted(time.isocalendar().week.unique()):
            m = np.logical_and(time.isocalendar().week == week, time.year == year)
            subset_data = df_proc[m].copy()
            
            # Map flights by unique flight number
            mapped_counts = get_mapped_counts(subset_data, "FlightNumber")
            df_proc.loc[subset_data.index, "FlightCount_week"] = mapped_counts

    return df_proc

def add_date_features(df_proc, time_col="ScheduleTime"):
    """Adds one-hot encoded time features from datetime column to df_proc"""

    time_series = df_proc[time_col]
    S = time_series.dt
    S_time = S.hour + (S.minute/60)

    df_time = pd.DataFrame({"t_dayofyear":S.day_of_year,
                            "t_dayofmonth":S.day,
                            "t_dayofweek":S.dayofweek,
                            "t_timeofday":S_time,
                            })

    df_out = pd.concat([df_proc, df_time], axis=1)
    df_out.drop(time_col, axis=1)

    return df_out

def add_cat_features(df_proc, cat_cols):
    """Adds one-hot encoded columns from cat_columns to df_proc"""
    
    dfs = []
    for col in cat_cols:
        df = pd.get_dummies(pd.Categorical(df_proc[col]))
        df.columns = [col + "_" + str(s) for s in list(df.columns.values)]
        dfs.append(df)

    df_onehot = pd.concat(dfs, axis=1)
    df_out = pd.concat([df_proc, df_onehot], axis=1)
    df_out = df_out.drop(cat_cols, axis=1)

    return df_out

def map_cat_as_numerical(df_proc, cat_cols, target_col):
    """ Maps categorical values to numerical by mean target value """

    targets = df_proc[target_col]

    for col in cat_cols:
        map_dict = {}

        uniq = df_proc[col].unique()
        for value in uniq:
            m = df_proc[col] == value
            delta = targets[m].mean() - targets[~m].mean()

            map_dict[value] = delta
            
        # Map values
        df_proc[col] = df_proc[col].map(map_dict)

    return df_proc

def map_cat_as_numerical_test(df_train, df_test, cat_cols, target_col):
    """ Maps categorical values to numerical by mean target value """

    targets = df_train[target_col]

    for col in cat_cols:
        map_dict = {}

        uniq = df_train[col].unique()
        for value in uniq:
            m = df_train[col] == value
            delta = targets[m].mean() - targets[~m].mean()

            map_dict[value] = delta

        # Map values
        df_test[col] = df_test[col].map(map_dict)

    return df_test

def normalize_minmax_cols(df_proc, norm_cols):
    """ Min-max normalizes norm_cols in df_proc """

    # Normalize
    for col in norm_cols:
        x = df_proc[col]
        x_norm = (x - x.min())
        x_norm = x_norm / x_norm.max()

        df_proc[col] = x_norm

    return df_proc

def add_time_delta(df_proc, start_time="2021-01-01", time_col="ScheduleTime"):
    """Adds column representing time proximity to end date in fractional months"""
    start_time = pd.to_datetime(start_time)

    df = df_proc[time_col]
    time_delta =  (df - start_time) / np.timedelta64(1, "M")
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
def create_trainval(df_proc, val_months=[12], val_years=[2021], time_col="ScheduleTime", y_col="LoadFactor"):
    """ Creates training and validation datasets by selected months/years """

    time = df_proc[time_col].dt

    # FloatingPointError
    m1 = time.year.isin(val_years)
    m2 = time.month.isin(val_months)
    mc = np.logical_and(m1, m2)

    val_idxs = df_proc[mc].index
    train_idxs = df_proc[~mc].index

    # Remove target y column and any other columns to exclude
    dataset_X = df_proc.drop([time_col, y_col], axis=1)
    dataset_y = df_proc[y_col]

    X_train, y_train = dataset_X.loc[train_idxs], dataset_y.loc[train_idxs]
    X_val, y_val = dataset_X.loc[val_idxs], dataset_y.loc[val_idxs]

    return X_train, y_train, X_val, y_val

# Machine learning
def create_test(df_proc, time_col="ScheduleTime"):
    """ Creates test dataset """

    # Remove target y column and any other columns to exclude
    dataset_X = df_proc.drop([time_col], axis=1)
    X_test = dataset_X

    return X_test

import sklearn.metrics as metrics
import scipy as sc
def test_performance_continuous(y_pred, y_true, text=""):
    # 
    mae = np.round(metrics.mean_absolute_error(y_pred, y_true), 3)
    mse = np.round(metrics.mean_squared_error(y_pred, y_true), 3)
    acc = np.round(1 - mae, 3)

    # Calculate pearson correlation beween predicted values (y_hat) and true values (y_true)
    pearson = sc.stats.pearsonr(y_true, y_pred)
    pearson, p = np.round(pearson[0], 3), pearson[1]

    print(f"{text} MAE: {mae}, MSE: {mse}, Pearson {pearson}, Acc: {acc}")

    return acc

def get_top_coef_perc(model, features, exclude_cols=[],
                    exclude_zero=True, perc=False,
                    verbose=0):
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
    if verbose:
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
    acc_train = test_performance_continuous(y_pred_train, y_train, text="Train:")

    y_pred_val = model.predict(X_val.values)
    acc_val = test_performance_continuous(y_pred_val, y_val, text="Valid:")

    exclude_cols = ["Aircraft", "Destination", "Airline"]
    exclude_cols = []
    
    try:
        get_top_coef_perc(model, X_train, exclude_cols, verbose=verbose)
    except Exception as E:
        print("{E}")

    return model, acc_train, acc_val, y_pred_val

def select_features(X_train, y_train, X_val, y_val, features):
    """ Selects given features in dataset """
    return X_train[features], y_train, X_val[features], y_val


def filter_features(X_train, y_train, X_val, y_val, filter_list):
    """ Filters features if string contains keywords in filter_list """

    features = X_train.columns
    for f in filter_list:
        features = features[~features.str.contains(f)]

    excluded_features = X_train.columns[~X_train.columns.isin(features)]
    print(f"Excluded {excluded_features}")
    

    X_train2, y_train2, X_val2, y_val2 = select_features(X_train, y_train, X_val, y_val, features)

    return X_train2, y_train2, X_val2, y_val2

