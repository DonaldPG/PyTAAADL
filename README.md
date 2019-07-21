# PyTAAADL

Tactical Asset Allocation using Deep Learning

a python project for practicing Deep Learning techniques using keras and tensorflow.

### Dependencies:

numpy pandas scipy tensorflow keras time datetime matplotlib

### Prerequisite:

create an hdf file that holds stock quotes

"python re-generateHDF5.py"

"python PyTAAADL_update_quotes_now.py"



### run PyTAAADL with the commands:

1. "python PyTAAADL__0_train_many_DL_models.py" -- Train many deep learning networks with a variety of randomly chosen parameters. Requires folder in repo named pngs
       
2. "python PyTAAADL__1_choose_best_performing_trained_models.py" -- Choose best performing DL models to carry forward for use in ensemble model (system of ensemble models). Requires folder in repo named pngs/best_performers4
         
3. "python PyTAAADL__2_create_dynamic_persistence_inputs.py" -- Compute ensemble models from amongst best performing DL models in folder pngs/best_performers4. Creates pandas dataframe for input to next step

4. "python PyTAAADL__3_evaluate_dynamic_persistence_inputs.py" -- Evaluate parameters to dynamically alter number_stocks, metric used to subset stocks from ensemble model, and number of months to persist choice of number_stocks, metrics.

5. "python PyTAAADL__5_suggest_stocks_and_investing_percentages.py" -- Evaluate parameters to dynamically alter number_stocks, metric used to subset stocks from ensemble model, and number of months to persist choice of number_stocks, metrics.


### re-evaluate after after the first trading day each month (22 trading days after the start of the previous month) with the commands:

A. "python PyTAAADL_update_quotes_now.py"         

B. "python PyTAAADL__2_create_dynamic_persistence_inputs.py" -- Compute ensemble models from amongst best performing DL models in folder pngs/best_performers4. Creates pandas dataframe for input to next step

C. "python PyTAAADL__5_suggest_stocks_and_investing_percentages.py" -- Evaluate parameters to dynamically alter number_stocks, metric used to subset stocks from ensemble model, and number of months to persist choice of number_stocks, metrics.
       
### PyTAAADL purpose

PyTAAADL recommends new stock holdings based on training using previous stock histories (DL data) combined with known stock gains/losses one month later (DL labels). PyTAAADL uses this trained network when run at the beginning of a month to suggest stocks to purchase and hold for one month. Repeat each month. It is currently untested but has been trained and backtested to suggest that chosen stocks will perform better than a market index such as the Nasdaq 100 index.

It's up to the user to decide if they want to do anything with the recommendations. This is designed and provided for entertainment only. The author does not accept and responsibility for anything done by others with the recommendations.

### Back-test performance

<div align="center">
  <img src="https://github.com/DonaldPG/PyTAAADL/blob/master/pngs/PyTAAADL_backtestWithTrend_2019-07-20.png">
</div>

### Cautionary Notes:

This is a work in progress. Back-test performance suggests unrealistic profits. I am still trying to locate inconsistencies that cause this. Simultaneously, I have been forward-testing since Feb 2018 to observe and benchmark performance against standard stock indices.

Backtests have terrific performance. Forward testing since February 2018, where this system is "paper-traded" manually, initially out-performed the Nasdaq 100 by as much as 10 percent. As of July 2019, it is underperforming the Nasdaq by about 10%. The code has been continually modified during this period. Automatic daily backtests also indicate that this period is one of under-performance that have also been observed, for example in 2000, 2008, and 2015. It is too soon to know if this is typical or uncharacteristic of performance in back-tests.

Backtest plots that start ca. 1991 contain different stocks for historical testing than those created by 're-generateHDF5.py' in the PyTAAA repository. Therefore backtest plots will not match those created by PyTAAADL. This is due to changes in the Nasdaq 100 index over time.

The backtest plots show only an approximation to "Buy & Hold" investing. This is particularly true for the Daily backtest that is created every time the PyTAAA code runs. Buy & Hold is approximated on the plot by the red value curves. The calculations assume that equal dollar investments are made in all the current stocks in the Nasdaq 100 index. For example, note that the current Nasdaq 100 stocks as of February 2014 did not have the same performance during 2000-2003 as the stocks in the index during 2000-2003. Whereas the Nasdaq Index lost more than 50% of its peak value, the stocks that are in the index as of February 2014 AND were also in the index in 2000, maintained nearly constant value over the period. Similar cautions need to be made about the historical backtest performance of PyTAAADL trading recommendations. Therefore, hypothetical performance as portrayed by PyTAAADL backtests should be viewed as untested and unverified. Actual investment performance under real market conditions will almost certainly be lower.
