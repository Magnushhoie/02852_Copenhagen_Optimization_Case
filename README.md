# 02852_Copenhagen_Optimization_Case
Case competition and project report for the [02582 Computational Data Analysis course](https://www.imm.dtu.dk/courses/02582/):

### [Forecasting for airports, Copenhagen Optimization case competition](https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/docs/Case%20for%2002582%20-%20Forecasting%20for%20airports.pdf)

This project build a Random Forest regressor to predict the relative fraction of occupied flight seats (Load Factor) for planned flights from the [Copenhagen Optimization](https://copenhagenoptimization.com/) flight dataset. The aim of the project is to build the best feature engineered dataset and model to minimize the mean absolute error for the March 2022 test set.


The training dataset comprises 39 449 flights between the time-period of 1st January 2021 to 28th February 2022. The features include the the flight scheduled calendar time including time of day, flight number, airline, destination aircraft type, flight type, geographical sector and seat capacity.  The test set consists of planned flights for March 2022, without the target load factor values to predict.

### See submitted [case_report_public.pdf](https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/case_report_public.pdf)

<img src="https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/figures/LOMO_pred.jpg" width=50% height=50%>

<img src="https://github.com/Magnushhoie/02852_Copenhagen_Optimization_Case/blob/main/figures/LOMO_true.jpg" width=50% height=50%>
