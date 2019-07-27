
"""
Created on Sat Dec  2 12:50:40 2017

@author: dp
"""

# --------------------------------------------------
# A Multilayer Perceptron implementation example using TensorFlow library.
# --------------------------------------------------

import os
import datetime
import numpy as np
import pandas
from keras import backend as K

from matplotlib import pyplot as plt


try:
    os.chdir(os.path.abspath(os.path.dir(__file__)))
except:
    os.chdir('C:\\Users\\don\\tf\\tf\\PyTAAADL_master\\PyTAAADL')

## local imports
_cwd = os.getcwd()
os.chdir(os.path.join(os.getcwd()))
_data_path = os.getcwd()
'''
from functions.quotes_for_list_adjClose import get_Naz100List, \
                                               arrayFromQuotesForList
'''
from functions.allstats import allstats
from functions.TAfunctions import _is_odd, \
                                  generateExamples, \
                                  generatePredictionInput, \
                                  generateExamples3layer, \
                                  generateExamples3layerGen, \
                                  generateExamples3layerForDate, \
                                  generatePredictionInput3layer, \
                                  get_params, \
                                  interpolate, \
                                  cleantobeginning, \
                                  cleantoend,\
                                  build_model, \
                                  get_predictions_input, \
                                  one_model_prediction, \
                                  ensemble_prediction

from functions.GetParams import GetParams

from functions.UpdateSymbols_inHDF5 import UpdateHDF_yf, \
                                           loadQuotes_fromHDF
import matplotlib.gridspec as gridspec

os.chdir(_cwd)

# --------------------------------------------------
# Get program parameters.
# --------------------------------------------------

run_params = GetParams()

# --------------------------------------------------
# set filename for datafram containing model persistence input data.
# --------------------------------------------------

persistence_hdf = os.path.join(_cwd,'pngs',run_params['folder_for_best_performers'],run_params['persistence_hdf'])
_performance_folder, persistence_hdf_fn = os.path.split(persistence_hdf)
_persistence_filename_prefix = os.path.splitext(persistence_hdf_fn)[0]

# set up pandas dataframe to hold results
df = pandas.DataFrame(columns=['dates', 'sort_modes', 'number_stocks', 'gains', 'symbols', 'weights', 'cumu_value'])

'''
df3 = pandas.HDFStore(persistence_hdf).select('table')

df3_array = df3.as_matrix()
df3_array_labels = np.array(df3.columns)
datearray_new_months = [df3_array[-1][0]]
'''

# --------------------------------------------------
# Import list of symbols to process.
# --------------------------------------------------

# read list of symbols from disk.
#stockList = 'Naz100'
stockList = run_params['stockList_predict']
filename = os.path.join(_data_path, 'symbols', 'Naz100_Symbols.txt')                   # plotmax = 1.e10, runnum = 902

# --------------------------------------------------
# Get quotes for each symbol in list
# process dates.
# Clean up quotes.
# Make a plot showing all symbols in list
# --------------------------------------------------

## update quotes from list of symbols
(symbols_directory, symbols_file) = os.path.split(filename)
basename, extension = os.path.splitext(symbols_file)
print((" symbols_directory = ", symbols_directory))
print(" symbols_file = ", symbols_file)
print("symbols_directory, symbols.file = ", symbols_directory, symbols_file)
###############################################################################################
do_update = False
if do_update == True:
    UpdateHDF_yf(symbols_directory, symbols_file)  ### assume hdf is already up to date
adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(filename)

firstdate = datearray[0]
#firstdate = df3_array[-5, 0]

# --------------------------------------------------
# Clean up missing values in input quotes
#  - infill interior NaN values using nearest good values to linearly interpolate
#  - copy first valid quote from valid date to all earlier positions
#  - copy last valid quote from valid date to all later positions
# --------------------------------------------------

for ii in range(adjClose.shape[0]):
    adjClose[ii, :] = cleantobeginning(adjClose[ii, :])
    adjClose[ii, :] = cleantoend(adjClose[ii, :])
    adjClose[ii, :] = interpolate(adjClose[ii, :])

print(" security values check: ", adjClose[np.isnan(adjClose)].shape)

gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
BH_value = 10000. * np.cumprod(gainloss,axis=1)
BH_value = BH_value.mean(axis=0)

# --------------------------------------------------
# prepare labeled data for DL training
# - set up historical data plus actual performance one month forward
# --------------------------------------------------

best_final_value = -99999
best_recent_final_value = -99999
num_periods_history = 20
first_history_index = 1500

try:
    for jdate in range(len(datearray)):
        year, month, day = datearray[jdate].split('-')
        datearray[jdate] = datetime.date(int(year), int(month), int(day))
except:
    pass

dates = []
company_number = []
first_day_of_month = []
new_month = []
previous_month = datearray[0]
for idate in range(adjClose.shape[1]):
    if idate == 0 or idate < first_history_index:
        beginning_of_month = False
    elif datearray[idate].month == datearray[idate-1].month:
        beginning_of_month = False
    else:
        beginning_of_month = True
    for icompany in range(adjClose.shape[0]):
        dates.append(datearray[idate])
        company_number.append(icompany)
        new_month.append(beginning_of_month)

datearray_new_months = []
for i,ii in enumerate(new_month):
    if ii == True:
        datearray_new_months.append(dates[i])

datearray_new_months = list(set(datearray_new_months))
datearray_new_months.sort()
datearray_new_months = [x for x in datearray_new_months if x >= firstdate ]

# --------------------------------------------------
# make predictions monthly for backtesting
# - apply multiple DL models and use 'num_stocks' most frequent stocks
# - break ties randomly
# --------------------------------------------------

#for model_filter in ['SP', 'Naz100', 'all']:
model_filter = run_params['model_filter']
model_filter = run_params['stockList']
models_folder = os.path.join(os.getcwd(), 'pngs', run_params['folder_for_best_performers'])
models_list = os.listdir(models_folder)
models_list = [i for i in models_list if '.txt' in i]
models_list = [i for i in models_list if 'bak' not in i]

if 'Naz' in model_filter:
    models_list = [i for i in models_list if 'Naz' in i]
elif 'SP' in model_filter:
    models_list = [i for i in models_list if 'SP' in i]
elif 'RU' in model_filter:
    models_list = [i for i in models_list if 'RU' in i]


# choices for sort_mode = 'sharpe', 'sortino', 'sharpe_plus_sortino'
sort_mode = run_params['sort_mode']

print("\n\n****************************\n")
print(" ... model_filter = ", model_filter)
print(" ... number of models found = ", len(models_list))

first_pass = True
'''
num_stocks_list = [5,6,7,8,9]
num_stocks_list = [2,3,4,5,6,7,8,9] # this should be used except for special cases
num_stocks_list = [3,5,7,9]
num_stocks_list = [10]
sort_mode_list = ['sortino', 'sharpe', 'count', 'equal'] # this should be used except for special cases
#sort_mode_list = ['sortino', 'count']
'''
# the following list should be used for 'num_stocks_list' and 'sort_mode_list',
# except in special cases
#  - [3,5,7,9]
#  - ['sortino', 'sharpe', 'count', 'equal']
num_stocks_list = run_params['num_stocks_list']
sort_mode_list = run_params['sort_mode_list']
# use all-inclusive choices so that subsets can be compared in subsequent computations (PyTAAADL__3,5)
#num_stocks_list = [2,3,4,5,6,7,8,9]                      # this should be used except for special cases
#sort_mode_list = ['sortino', 'sharpe', 'count', 'equal'] # this should be used except for special cases

for inum_stocks in num_stocks_list:

    print(" ... inum_stocks = ", inum_stocks)

    # -------------------------------------------------------------------------
    # initialize performance history for individual models
    # -------------------------------------------------------------------------
    cumu_models = []
    for im, imodel in enumerate(models_list):
        cumu_models.append([10000.0])

    # -------------------------------------------------------------------------
    # compute performance history for individual models
    # -------------------------------------------------------------------------

    for im, imodel in enumerate(models_list):
        avg_gain, _, _, models_plotdates = one_model_prediction(os.path.abspath(os.path.join(models_folder,imodel)),
                                                         first_history_index,
                                                         datearray,
                                                         adjClose,
                                                         symbols,
                                                         inum_stocks)
        cumu_models[im] = avg_gain
        K.clear_session()


    # -------------------------------------------------------------------------
    # initialize performance history for ensemble model
    # -------------------------------------------------------------------------
    cumu_system = [10000.0]
    cumu_system_worst = [10000.0]
    cumu_BH = [10000.0]
    cumu_dynamic_system = [10000.0]
    cumu_dynamic_reversion_system = [10000.0]
    plotdates = [datearray_new_months[0]]
    _weights_stdev = [0]

    _forecast_mean = []
    _forecast_median = []
    _forecast_stdev = []

    # -------------------------------------------------------------------------
    # compute performance for all models collectively, as ensemble model
    # -------------------------------------------------------------------------

    #for i, idate in enumerate(datearray_new_months[:-1]):
    for i, idate in enumerate(datearray_new_months[:-1]):

        recent_comparative_gain = [1.]
        recent_comparative_month_gain = [1.]
        recent_comparative_method = ['cash']
        for sort_mode in sort_mode_list:

            if first_pass:
                # try to open existing dataframe. Create new one if unsuccessful.
                try:
                    #df = pandas.HDFStore(persistence_hdf).select('table')
                    #print("...found existing dataframe and loaded data")
                    #print("...name of existing persistence_hdf = ", persistence_hdf)
                    df3 = pandas.HDFStore(persistence_hdf).select('table')
                    #df = pandas.DataFrame(columns=['dates', 'sort_modes', 'number_stocks', 'gains', 'symbols', 'weights', 'cumu_value'])
                    print("...opened existing dataframe")
                except:
                    # set up pandas dataframe to hold results
                    df3 = pandas.DataFrame(columns=['dates', 'sort_modes', 'number_stocks', 'gains', 'symbols', 'weights', 'cumu_value'])
                    print("...created new dataframe")
                first_pass = False

            # compute only if (inum_stocks, date, sort_mode) don't already exists in dataframe
            print("    ... inum_stocks = ", inum_stocks, "idate = ", idate, 'sort_mode = ', sort_mode)
            test_subset = df3.loc[df3['number_stocks'].isin([inum_stocks]) & df3['dates'].isin([idate]) & df3['sort_modes'].isin([sort_mode])]
            #print("...test_subset = ", test_subset, "\n\n")

            if test_subset.size == 0:

                if sort_mode == sort_mode_list[0]:
                    print("")

                #prediction_date_index = datearray.index(idate)
                avg_gain, BH_gain, sorted_symbols, symbols_weights = ensemble_prediction(models_folder,
                                                                                         models_list,
                                                                                         idate,
                                                                                         datearray, #[:prediction_date_index+1],
                                                                                         adjClose, #[:,:prediction_date_index+1],
                                                                                         symbols,
                                                                                         inum_stocks,
                                                                                         sort_mode=sort_mode)

                if symbols_weights[np.isnan(symbols_weights)].shape[0] > 0 or inum_stocks==0:
                    avg_gain = 0.
                    symbols_weights = np.ones(symbols_weights.shape, 'float')

                """
                if first_pass:
                    # try to open existing dataframe. Create new one if unsuccessful.
                    try:
                        df = pandas.HDFStore(persistence_hdf).select('table')
                    except:
                        # set up pandas dataframe to hold results
                        df = pandas.DataFrame(columns=['dates', 'sort_modes', 'number_stocks', 'gains', 'symbols', 'weights', 'cumu_value'])
                    for iinum_stocks in num_stocks_list:
                        for isort_mode in sort_mode_list:
                            datarow = [ datearray[first_history_index], isort_mode, iinum_stocks, 0., [], [], 10000.]
                            df.loc[len(df)] = datarow
                    datarow = [ idate, sort_mode, inum_stocks, avg_gain, sorted_symbols, symbols_weights, np.isnan]
                    df.loc[len(df)] = datarow
                    smode = df.values[-1,1]
                    nstocks = df.values[-1,2]
                    indices = df.loc[np.logical_and(df['sort_modes']==smode,df['number_stocks']==nstocks)]['gains'].index
                    method_cumu_gains = 10000.*(df.values[indices,-4]+1.).cumprod()
                    df.loc[len(df)-1,'cumu_value'] = method_cumu_gains[-1]
                    first_pass = False
                else:
                    datarow = [ idate, sort_mode,
                             inum_stocks, avg_gain,
                             sorted_symbols, symbols_weights, np.isnan]
                    df.loc[len(df)] = datarow
                    smode = df.values[-1,1]
                    nstocks = df.values[-1,2]
                    indices = df.loc[np.logical_and(df['sort_modes']==smode,df['number_stocks']==nstocks)]['gains'].index
                    method_cumu_gains = 10000.*(df.values[indices,-4]+1.).cumprod()
                    df.loc[len(df)-1,'cumu_value'] = method_cumu_gains[-1]
                """

                # write info to last row in ensemble model persistence dataframe
                datarow = [ idate, sort_mode,
                         inum_stocks, avg_gain,
                         sorted_symbols, symbols_weights, np.isnan]
                df.loc[len(df)] = datarow

                smode = df.values[-1,1]
                nstocks = df.values[-1,2]
                indices = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]['gains'].index
                method_cumu_gains = 10000.*(df.values[indices,-4]+1.).cumprod()
                df.loc[len(df)-1,'cumu_value'] = method_cumu_gains[-1]

            else:
                print('       ... use existing datarow ')
                #print('    ... use existing datarow ',(inum_stocks, sort_mode, idate))
                datarow = test_subset
                #print('    ... datarow = ',datarow)
                '''
                smode = datarow['sort_modes'].values[0]
                nstocks = datarow['number_stocks'].values[0]
                avg_gain = datarow['gains'].values[0]
                indices = df.loc[np.logical_and(df['sort_modes']==smode,df['number_stocks']==nstocks)]['gains'].index
                '''
                df = df.append(datarow, ignore_index=True)
                indices = np.arange(len(df))
                #method_cumu_gains = 10000.*(df.values[indices,-4]+1.).cumprod()
                try:
                    
                    method_cumu_gains = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]['cumu_value'].values
                    avg_gain =          df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]['gains'].values[-1]
                    symbols_weights = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]['weights'].values[-1]
                    sorted_symbols = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]['symbols'].values[-1]
                    '''
                    _method_cumu_gains = df.loc[df['number_stocks'].isin([inum_stocks]) & df['dates'].isin([idate]) & df['sort_modes'].isin([sort_mode])]['cumu_value'].values
                    avg_gain =          df.loc[df['number_stocks'].isin([inum_stocks]) & df['dates'].isin([idate]) & df['sort_modes'].isin([sort_mode])]['gains'].values[-1]
                    symbols_weights =   df.loc[df['number_stocks'].isin([inum_stocks]) & df['dates'].isin([idate]) & df['sort_modes'].isin([sort_mode])]['weights'].values[-1]
                    sorted_symbols =    df.loc[df['number_stocks'].isin([inum_stocks]) & df['dates'].isin([idate]) & df['sort_modes'].isin([sort_mode])]['symbols'].values[-1]
                    '''

                except:
                    avg_gain = 0.0
                    symbols_weights = np.ones(inum_stocks, 'float') / float(inum_stocks)
                '''
                try:
                    date2_index = len(df)
                    avg_gain = df['gains'][date2_index-1]
                    symbols_weights = df['weights'][date2_index-1]
                    sorted_symbols = df['symbols'][date2_index-1]
                except:
                    avg_gain = 0.0
                    symbols_weights = np.ones(inum_stocks, 'float') / float(inum_stocks)
                '''
                try:
                    # index for beginning of current month
                    date2_index = datearray.index(idate)
                    # index for beginning of previous month
                    date1_index = datearray.index(datearray_new_months[max(0, i-1)])
                    BH_gain = BH_value[date2_index] / BH_value[date1_index] - 1.
                    print('       ... B&H date1, date2 = ', datearray[date1_index], datearray[date2_index])
                except:
                    BH_gain = 0.0
                
                """
                plotdates = []
                plotvalues = []
                cumu_BH = []
                for i, idate in enumerate(datearray_new_months[:-1]):
                    # index for beginning of current month
                    date2_index = datearray.index(idate)
                    # index for beginning of previous month
                    date1_index = datearray.index(datearray_new_months[max(0, i-1)])
                    BH_gain = BH_value[date2_index] / BH_value[date1_index] - 1.
                    plotdates.append(datearray[date2_index])
                    plotvalues.append(BH_value[date2_index] * 10000. / BH_value[datearray.index(datearray_new_months[0])])
                    if i == 0:
                        cumu_BH.append(10000.)
                    else:
                        cumu_BH.append(cumu_BH[-1] * (1. + BH_gain))
                    print('       ... B&H date1, date2, BH_value = ',
                          datearray[date1_index],
                          datearray[date2_index],
                          BH_value[date2_index] * 10000. / BH_value[datearray.index(datearray_new_months[0])])
                """


            print("       ... avg_gain = ", avg_gain, " BH_gain = ", BH_gain)
            cumu_system.append(cumu_system[-1] * (1.+avg_gain))
            recent_comparative_month_gain.append(1.+avg_gain)
            if sort_mode == sort_mode_list[0]:
                cumu_BH.append(cumu_BH[-1] * (1.+BH_gain))
            plotdates.append(idate)
            _weights_stdev.append(symbols_weights.std())

            print("       ... system, B&H = ",
                  idate,
                  format(avg_gain, '3.1%'), format(BH_gain, '3.1%'),
                  sort_mode, format(method_cumu_gains[-1], '10,.0f'), format(cumu_BH[-1], '10,.0f'))
            try:
                print("       ... symbols, weights = ",
                  sorted_symbols, np.around(symbols_weights,3),
                  symbols_weights.sum(),
                  #format(method_cumu_gains[-1]/method_cumu_gains[-2],'5.3f') )
                  format(1.+avg_gain, '5.3f') )
            except:
                print("       ... symbols, weights = ",
                  sorted_symbols, np.around(symbols_weights,3), symbols_weights.sum() )

            '''
            if len(df)>1 and len(df)%20==0:
                #print(" ...datarow = ", datarow)
                print(" ...df (partial) = ", df.values[:,np.array([0,1,2,3,6])])
            '''

            try:
                #recent_comparative_gain.append(method_cumu_gains[-2]/method_cumu_gains[-5])
                recent_comparative_gain.append(1.+avg_gain)
                recent_comparative_method.append(sort_mode)
            except:
                recent_comparative_gain.append(1.+BH_gain)
                recent_comparative_method.append('BH')
            if sort_mode == sort_mode_list[-1]:
                best_comparative_index = np.argmax(recent_comparative_gain)
                worst_comparative_index = np.argmin(recent_comparative_gain[1:])+1
                cumu_dynamic_system.append(cumu_dynamic_system[-1] * recent_comparative_month_gain[best_comparative_index])
                cumu_dynamic_reversion_system.append(cumu_dynamic_reversion_system[-1] * recent_comparative_month_gain[worst_comparative_index])
                print("       ... methods, near-term gains ", recent_comparative_method, np.around(recent_comparative_gain,2))
                print("       ... dynamic system (best, worst) = ",
                      recent_comparative_method[best_comparative_index],
                      format(cumu_dynamic_system[-1], '10,.0f'),
                      recent_comparative_method[worst_comparative_index],
                      format(cumu_dynamic_reversion_system[-1], '10,.0f'),
                      "\n")

    print(" ...system, B&H = ", format(cumu_system[-1], '10,.0f'), format(cumu_BH[-1], '10,.0f'), "\n\n")

    # write comparative data to hdf
    #df.set_index('dates', inplace=True)
    #df.to_hdf(os.path.join(os.getcwd(),'persistence_data_full_v3.hdf'), 'table', table=True, mode='a')
    #df.to_hdf(persistence_hdf, 'table3', table=True, mode='a')
    store = pandas.HDFStore(persistence_hdf)
    store['table'] = df
    store.close()

# plot results over full timeframe
os.chdir(_cwd)
plt.close(2)
subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,2])
plt.figure(2, figsize=(14, 10))
plt.clf()
plt.subplot(subplotsize[0])
plt.plot(models_plotdates, cumu_BH, 'r-', lw=3, label='B&H')
plt.grid(True)
plt.yscale('log')
colors = ['b','palegreen','y','m']
for i, inum_stocks in enumerate(num_stocks_list):
    for j, sort_mode in enumerate(sort_mode_list):
        system_label = 'ensemble '+sort_mode+' '+str(inum_stocks)
        df_subset = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]
        plt.plot(df_subset['dates'], df_subset['cumu_value'], c=colors[j], lw=.5*(i+1), label=system_label)
for im, imodel in enumerate(models_list):
    model_label = imodel.replace('.txt','')
    model_label = model_label[1:]
    plt.plot(models_plotdates, cumu_models[im], 'k-', label=model_label, lw=3.)
    print(model_label)
plt.legend(fontsize=8)
plt.title('Best ensemble systems\nPredict on stocks in '+stockList)
plt.subplot(subplotsize[1])
plt.grid(True)
plt.title('weights stdev')
for i, inum_stocks in enumerate(num_stocks_list):
    for j, sort_mode in enumerate(sort_mode_list):
        system_label = 'ensemble '+sort_mode+' '+str(inum_stocks)
        df_subset = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]
        __weights = df_subset['weights']
        _weights_std = []
        for iweights in __weights:
            _weights_std.append(np.std(iweights))
        plt.plot(df_subset['dates'], _weights_std, c=colors[j], lw=.5*(i+1), label=system_label)
plt.savefig(os.path.join(models_folder, model_filter+"_ensemble_best_fig-2"+'.png'), format='png')


# plot results for medium timeframe
num_months = int(12*13.75)
plt.close(3)
subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,2])
plt.figure(3, figsize=(14, 10))
plt.clf()
plt.subplot(subplotsize[0])
plt.plot(models_plotdates[-num_months:], cumu_BH[-num_months:]/cumu_BH[-num_months]*10000., 'r-', lw=3, label='B&H')
plt.grid(True)
plt.yscale('log')
colors = ['b','palegreen','y','m']
for i, inum_stocks in enumerate(num_stocks_list):
    for j, sort_mode in enumerate(sort_mode_list):
        system_label = 'ensemble '+sort_mode+' '+str(inum_stocks)
        df_subset = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]
        yvals = np.array(df_subset['cumu_value'])
        yvals = yvals[-num_months:]/yvals[-num_months]*10000.
        plt.plot(df_subset['dates'][-num_months:], yvals, c=colors[j], lw=.5*(i+1), label=system_label)
for im, imodel in enumerate(models_list):
    model_label = imodel.replace('.txt','')
    model_label = model_label[1:]
    yvals = cumu_models[im][-num_months:]/cumu_models[im][-num_months]*10000.
    plt.plot(models_plotdates[-num_months:], yvals, 'k-', label=model_label, lw=3.)
    print(model_label)
plt.legend(fontsize=8)
plt.title('Best ensemble systems\nPredict on stocks in '+stockList)
plt.subplot(subplotsize[1])
plt.grid(True)
plt.title('weights stdev')
for i, inum_stocks in enumerate(num_stocks_list):
    for j, sort_mode in enumerate(sort_mode_list):
        system_label = 'ensemble '+sort_mode+' '+str(inum_stocks)
        df_subset = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]
        __weights = df_subset['weights']
        _weights_std = []
        for iweights in __weights:
            _weights_std.append(np.std(iweights))
        plt.plot(df_subset['dates'][-num_months:], _weights_std[-num_months:], c=colors[j], lw=.5*(i+1), label=system_label)
plt.savefig(os.path.join(models_folder, model_filter+"_ensemble_best_fig-3"+'.png'), format='png')


# plot results for recent timeframe
num_months = int(12*4.75)
plt.close(4)
subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,2])
plt.figure(4, figsize=(14, 10))
plt.clf()
plt.subplot(subplotsize[0])
plt.plot(models_plotdates[-num_months:], cumu_BH[-num_months:]/cumu_BH[-num_months]*10000., 'r-', lw=3, label='B&H')
plt.grid(True)
plt.yscale('log')
colors = ['b','palegreen','y','m']
for i, inum_stocks in enumerate(num_stocks_list):
    for j, sort_mode in enumerate(sort_mode_list):
        system_label = 'ensemble '+sort_mode+' '+str(inum_stocks)
        df_subset = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]
        yvals = np.array(df_subset['cumu_value'])
        yvals = yvals[-num_months:]/yvals[-num_months]*10000.
        plt.plot(df_subset['dates'][-num_months:], yvals, c=colors[j], lw=.5*(i+1), label=system_label)
for im, imodel in enumerate(models_list):
    model_label = imodel.replace('.txt','')
    model_label = model_label[1:]
    yvals = cumu_models[im][-num_months:]/cumu_models[im][-num_months]*10000.
    plt.plot(models_plotdates[-num_months:], yvals, 'k-', label=model_label, lw=3.)
    print(model_label)
plt.legend(fontsize=18)
plt.title('Best ensemble systems\nPredict on stocks in '+stockList)
plt.subplot(subplotsize[1])
plt.grid(True)
plt.title('weights stdev')
for i, inum_stocks in enumerate(num_stocks_list):
    for j, sort_mode in enumerate(sort_mode_list):
        system_label = 'ensemble '+sort_mode+' '+str(inum_stocks)
        df_subset = df.loc[np.logical_and(df['sort_modes']==sort_mode,df['number_stocks']==inum_stocks)]
        __weights = df_subset['weights']
        _weights_std = []
        for iweights in __weights:
            _weights_std.append(np.std(iweights))
        plt.plot(df_subset['dates'][-num_months:], _weights_std[-num_months:], c=colors[j], lw=.5*(i+1), label=system_label)
plt.savefig(os.path.join(models_folder, model_filter+"_ensemble_best_fig-4"+'.png'), format='png')

num_stocks = list(set(list(df['number_stocks'])))
start_date = datetime.date(1997,1,1)
end_date = datetime.date(df['dates'].max().year+1,1,1)
max_y = np.around(df['cumu_value'].max()*1.15,-8)
colors = ['b', 'g', 'y', 'c', 'v']
for inum_stocks in num_stocks:
    plt.close(5)
    plt.figure(5, figsize=(14, 10))
    plt.clf()
    plt.title('Ensemble System Components\nnumber_stocks = '+str(inum_stocks), fontsize=20)
    plt.grid(True)
    plt.yscale('log')
    plt.xlim((start_date, end_date))
    plt.ylim((10000., max_y))
    index = (np.abs(np.array(datearray)-datetime.date(1997,1,1))).argmin()
    factor = 10000. / BH_value[index]

    modes = list(set(list(df['sort_modes'])))
    for i, imode in enumerate(modes):
        df_subset = df.loc[np.logical_and(df['sort_modes']==imode,df['number_stocks']==inum_stocks)]
        plt.plot( df_subset['dates'], df_subset['cumu_value'], '-', color=colors[i], label=imode )
    plt.plot(datearray,BH_value*factor,'r-',lw=2.5,label='B & H')
    plt.legend(fontsize=16)
    plt.xlabel('date', fontsize=16)
    plt.ylabel('Portfolio Value $', fontsize=16)
    plt.text(datetime.date(df['dates'].min().year+1,1,1), 7000., 'ensemble data: '+persistence_hdf)
    plt.text(datetime.date(df['dates'].min().year+1,1,1), 5000., 'final date: '+str(df['dates'].max()))
    plt.text(datetime.date(df['dates'].min().year+1,1,1), 3500., 'updated: '+str(datetime.datetime.now()))
    plt.savefig(os.path.join(_performance_folder, 'EnsembleSystemComponentValue_'+str(inum_stocks)+'.png'), format='png')
