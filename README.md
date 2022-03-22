# 02852_Copenhagen_Optimization_Case
Case competition and project report for the [02582 Computational Data Analysis course](https://www.imm.dtu.dk/courses/02582/):

### [Forecasting for airports, Copenhagen Optimization case competition](https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/docs/Case%20for%2002582%20-%20Forecasting%20for%20airports.pdf)

This project build a Random Forest regressor to predict the relative fraction of occupied flight seats (Load Factor) for planned flights from the [Copenhagen Optimization](https://copenhagenoptimization.com/) flight dataset. The aim of the project is to build the best feature engineered dataset and model to minimize the mean absolute error for the March 2022 test set.


The training dataset comprises 39 449 flights between the time-period of 1st January 2021 to 28th February 2022. The features include the the flight scheduled calendar time including time of day, flight number, airline, destination aircraft type, flight type, geographical sector and seat capacity.  The test set consists of planned flights for March 2022, without the target load factor values to predict.

### Feature engineering
Categorical values are mapped to continuous values by calculating the mean target value when the categorical value is present vs not present. This is described in more detail in the report.

```python
# Iterate through categorical feature columns
for col in cat_cols:
  map_dict = {}
  
  # Extract unique categorical values for this feature # E.g. flight numbers 899, 903 and E21
  uniq = df_proc[col].unique()
  
  # For each unique value, calculate the mean target value when the categorical value is present and not present
  for value in uniq:

    # A mask allows us to select a subset or everything but the subset of the data
    mask = df_proc[col] == value
    
    # Calculate difference between the two groups
    delta = targets[mask].mean() - targets[~mask].mean()
    
    # Add value to a mapping dict
    map_dict[value] = delta
    
    # Use dict to map original feature values to the new numerical values
    df_proc[col] = df_proc[col].map(map_dict)
```

### See submitted [case_report_public.pdf](https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/case_report_public.pdf)

### See [Jupyter notebook](https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/notebooks/mh_1_final.ipynb) with code

<img src="https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/figures/LOMO_pred.jpg" width=50% height=50%>

<img src="https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/figures/LOMO_true.jpg" width=50% height=50%>

<img src="https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/figures/LOMO_pred_dist.jpg" width=50% height=50%>

