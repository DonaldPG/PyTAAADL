# PyTAAADL

Tactical Asset Allocation using Deep Learning

a python project for practicing Deep Learning techniques using keras and tensorflow.

Dependencies:

numpy scipy tensorflow keras time datetime matplotlib

Usage:

create an hdf file that holds stock quotes (as in repository pyTAAA)

"python re-generateHDF5.py"

edit the file containing information for PyTAA to report results

options exist to send an email to your desired email from another email account (they don't have to be the same email account)
edit PyTAAA.params with a text editor and replace example values with your information
run PyTAAADL with the command: "python SP500_DL_3layer_DropRandomStocks_monthly_sensitivityTest.py"

PyTAAADL recommends new stock holdings based on training using previous stock histories (DL data) combined with known stock gains/losses one month later (DL labels). PyTAAADL uses this trained network to when run at the beginning of a month to suggest stocks to purchase and hold for one month. Repeat each month. It is currently untested but has been trained and backtested to suggest that chosen stocks will perform better than a market index such as the Nasdaq 100 index.

It's up to the user to decide if they want to do anything with the recommendations. This is designed and provided for entertainment only. The author does not accept and responsibility for anything done by others with the recommendations.

Notes:

Backtest plots that start ca. 1991 contain different stocks for historical testing than those created by 're-generateHDF5.py' in the PyTAAA repository. Therefore backtest plots will not match those created by PyTAAADL. This is due to changes in the Nasdaq 100 index over time.

The backtest plots show only an approximation to "Buy & Hold" investing. This is particularly true for the Daily backtest that is created every time the PyTAAA code runs. Buy & Hold is approximated on the plot by the red value curves. The calculations assume that equal dollar investments are made in all the current stocks in the Nasdaq 100 index. For example, note that the current Nasdaq 100 stocks as of February 2014 did not have the same performance during 2000-2003 as the stocks in the index during 2000-2003. Whereas the Nasdaq Index lost more than 50% of its peak value, the stocks that are in the index as of February 2014 AND were also in the index in 2000, maintained nearly constant value over the period. Similar cautions need to be made about the historical backtest performance of PyTAAA trading recommendations. Therefore, hypothetical performance as portrayed by PyTAAADL backtests should be viewed as untested and unverified. Actual investment performance under real market conditions will almost certainly be lower.
current stocks it can choose match the index.
