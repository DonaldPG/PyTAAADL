
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
import configparser

from keras import backend as K
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam, Adagrad, Nadam

from matplotlib import pyplot as plt

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
                                  fix_params_file, \
                                  get_params, \
                                  interpolate, \
                                  cleantobeginning, \
                                  cleantoend
from functions.GetParams import GetParams
from functions.UpdateSymbols_inHDF5 import UpdateHDF5, \
                                           loadQuotes_fromHDF
os.chdir(_cwd)

"""
def fix_params_file(config_filename, verbose=False):

    current_folder = os.getcwd()

    fn_path, fn = os.path.split(config_filename)

    os.chdir(fn_path)
    config_filename_bak = fn.replace(".txt",".bak.txt")

    if os.path.isfile(config_filename_bak):
        return

    import shutil
    #print("config_filename = ", fn)
    #print("backed-up config_filename_bak = ", config_filename_bak)
    shutil.copy(fn, config_filename_bak)

    with open(config_filename_bak,'r') as f:
        text = f.read()
        text_list = text.split("\n")
        previous_labels = []
        edited_text_list = []
        for i, line in enumerate(text_list):
            label = line.split(":")[0]
            if label not in previous_labels:
                edited_text_list.append(line+"\n")
            previous_labels.append(label)

    with open(fn,'w') as f:
        for text in edited_text_list:
            f.write(text)

    os.chdir(current_folder)
    if verbose:
        print("    ... params file successfully processed for duplicate keys")

    return


def get_params(config_filename):

    # --------------------------------------------------
    # Input parameters
    # --------------------------------------------------

    fix_params_file(config_filename)

    parser = configparser.ConfigParser()
    #configfile = open(config_filename, "r")
    parser.read(config_filename)

    try:
        perform_batch_normalization = parser.get("training_params", "perform_batch_normalization")
    except:
        perform_batch_normalization = True

    try:
        use_dense_layers = parser.get("training_params", "use_dense_layers")
    except:
        use_dense_layers = False

    try:
        feature_map_factor = parser.get("training_params", "feature_map_factor")
    except:
        feature_map_factor = 1

    try:
        loss_function = parser.get("training_params", "loss function")
    except:
        loss_function = 'mse'

    try:
        optimizer_choice = parser.get("training_params", "optimizer_choice")
    except:
        optimizer_choice = 'RMSprop'

    use_leaky_relu = parser.get("training_params", "use_leaky_relu")
    leaky_relu_alpha = parser.get("training_params", "leaky_relu_alpha")

    num_stocks = parser.get("training_params", "num_stocks")
    increments = parser.get("training_params", "increments")
    num_periods_history = parser.get("training_params", "num_periods_history")
    first_history_index = parser.get("training_params", "first_history_index")
    _sharpe_ratio_system = parser.get("training_params", "_sharpe_ratio_system")
    _sharpe_ratio_recent_system = parser.get("training_params", "_sharpe_ratio_recent_system")

    weights_filename = parser.get("training_params", "weights_filename")
    model_json_filename = parser.get("training_params", "model_json_filename")

    # put params in a dictionary
    params = {}
    params['perform_batch_normalization'] = perform_batch_normalization
    params['use_dense_layers'] = use_dense_layers
    params['use_leaky_relu'] = use_leaky_relu
    params['leaky_relu_alpha'] = float(leaky_relu_alpha)

    params['feature_map_factor'] = int(feature_map_factor)
    params['optimizer_choice'] = optimizer_choice
    params['loss_function'] = loss_function
    params['num_stocks'] = int(num_stocks)
    _increments = increments.replace(" ",",").replace("[,","[").replace(",,",",")
    params['increments'] = eval(_increments)
    params['num_periods_history'] = int(num_periods_history)
    params['first_history_index'] = int(first_history_index)
    params['_sharpe_ratio_system'] = float(_sharpe_ratio_system)
    params['_sharpe_ratio_recent_system'] = float(_sharpe_ratio_recent_system)

    params['weights_filename'] = weights_filename
    params['model_json_filename'] = model_json_filename

    return params
"""

def get_predictions_input(config_filename, adjClose, datearray):

    params = get_params(config_filename)

    first_history_index = params['first_history_index']
    num_periods_history = params['num_periods_history']
    increments = params['increments']

    print(" ... generating examples ...")
    Xpredict, Ypredict, dates_predict, companies_predict = generateExamples3layerGen(datearray,
                                               adjClose,
                                               first_history_index,
                                               num_periods_history,
                                               increments,
                                               output_incr='monthly')

    print(" ... examples generated ...")
    return Xpredict, Ypredict, dates_predict, companies_predict


def build_model(config_filename, verbose=False):
    # --------------------------------------------------
    # build DL model
    # --------------------------------------------------

    params = get_params(config_filename)

    optimizer_choice = params['optimizer_choice']
    loss_function = params['loss_function']

    weights_filename = params['weights_filename']
    model_json_filename = params['model_json_filename']

    # load model and weights from json and hdf files.
    json_file = open(model_json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weights_filename)
    if verbose:
        print("    ... model successfully loaded from disk")

    if optimizer_choice == 'RMSprop':
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    elif optimizer_choice == 'Adam':
        optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif optimizer_choice == 'Adagrad':
        optimizer = Adagrad(lr=0.005, epsilon=1e-08, decay=0.0)
    elif optimizer_choice == 'Nadam':
        optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    if verbose:
        model.summary()
    model.compile(optimizer=optimizer, loss=loss_function)

    return model


def one_model_prediction(imodel, first_history_index, datearray, adjClose, symbols, num_stocks, verbose=False):

    # --------------------------------------------------
    # build DL model
    # --------------------------------------------------

    config_filename = imodel.replace('.hdf','.txt')
    print("\n ... config_filename = ", config_filename)
    #print(".", end='')
    model = build_model(config_filename)

    # collect meta data for weighting ensemble_symbols
    params = get_params(config_filename)
    #num_stocks = params['num_stocks']
    num_periods_history = params['num_periods_history']
    increments = params['increments']

    symbols_predict = symbols
    Xpredict, Ypredict, dates_predict, companies_predict = generateExamples3layerGen(datearray,
                                                                                  adjClose,
                                                                                  first_history_index,
                                                                                  num_periods_history,
                                                                                  increments,
                                                                                  output_incr='monthly')

    dates_predict = np.array(dates_predict)
    companies_predict = np.array(companies_predict)

    # --------------------------------------------------
    # make predictions monthly for backtesting
    # - there might be some bias since entire preiod
    #   has data used for training
    # --------------------------------------------------

    try:
        model.load_weights(imodel)
    except:
        pass

    dates_predict = np.array(dates_predict)
    companies_predict = np.array(companies_predict)

    inum_stocks = num_stocks
    cumu_system = [10000.0]
    cumu_BH = [10000.0]
    plotdates = [dates_predict[0]]
    _forecast_mean = []
    _forecast_median = []
    _forecast_stdev = []
    for i, idate in enumerate(dates_predict[1:]):
        if idate != dates[-1] and companies_predict[i] < companies_predict[i-1]:
            # show predictions for (single) last date
            _Xtrain = Xpredict[dates_predict == idate]
            _dates = np.array(dates_predict[dates_predict == idate])
            _companies = np.array(companies_predict[dates_predict == idate])
            #print("forecast shape = ", model.predict(_Xtrain).shape)
            _forecast = model.predict(_Xtrain)[:, 0]
            _symbols = np.array(symbols_predict)

            indices = _forecast.argsort()
            sorted_forecast = _forecast[indices]
            sorted_symbols = _symbols[indices]

            try:
                _Ytrain = Ypredict[dates_predict == idate]
                sorted_Ytrain = _Ytrain[indices]
                BH_gain = sorted_Ytrain.mean()
            except:
                BH_gain = 0.0

            avg_gain = sorted_Ytrain[-inum_stocks:].mean()

            _forecast_mean.append(_forecast.mean())
            _forecast_median.append(np.median(_forecast))
            _forecast_stdev.append(_forecast.std())

            if verbose:
                print(" ... date, system_gain, B&H_gain = ",
                      idate,
                      format(avg_gain, '3.1%'), format(BH_gain, '3.1%'),
                      sorted_symbols[-inum_stocks:])
            cumu_system.append(cumu_system[-1] * (1.+avg_gain))
            cumu_BH.append(cumu_BH[-1] * (1.+BH_gain))
            plotdates.append(idate)
    print(" ... system, B&H = ", format(cumu_system[-1], '10,.0f'), format(cumu_BH[-1], '10,.0f'))

    return cumu_system, cumu_BH, sorted_symbols, plotdates



def ensemble_prediction(models_list, idate, datearray, adjClose, num_stocks, sort_mode='sharpe'):
    #--------------------------------------------------------------
    # loop through best models and pick companies from ensemble prediction
    #--------------------------------------------------------------

    ensemble_symbols = []
    ensemble_Ytrain = []
    ensemble_sharpe = []
    ensemble_recent_sharpe = []
    ensemble_equal = []
    ensemble_rank = []
    for iii,imodel in enumerate(models_list):

        # --------------------------------------------------
        # build DL model
        # --------------------------------------------------

        config_filename = os.path.join(models_folder, imodel).replace('.hdf','.txt')
        #print(" ... config_filename = ", config_filename)
        print(".", end='')
        model = build_model(config_filename, verbose=False)

        # collect meta data for weighting ensemble_symbols
        params = get_params(config_filename)

        #num_stocks = params['num_stocks']
        num_periods_history = params['num_periods_history']
        increments = params['increments']

        symbols_predict = symbols
        Xpredict, Ypredict, dates_predict, companies_predict = generateExamples3layerForDate(idate,
                                                                                             datearray,
                                                                                             adjClose,
                                                                                             num_periods_history,
                                                                                             increments,
                                                                                             output_incr='monthly',
                                                                                             verbose=False)

        dates_predict = np.array(dates_predict)
        companies_predict = np.array(companies_predict)

        # --------------------------------------------------
        # make predictions monthly for backtesting
        # - there might be some bias since entire preiod
        #   has data used for training
        # --------------------------------------------------

        weights_filename = os.path.join(models_folder, imodel)
        try:
            model.load_weights(weights_filename)
        except:
            pass

        # show predictions for (single) last date
        _Xtrain = Xpredict[dates_predict == idate]
        _Ytrain = Ypredict[dates_predict == idate][:,0]
        _dates = np.array(dates_predict[dates_predict == idate])
        _companies = np.array(companies_predict[dates_predict == idate])
        _forecast = model.predict(_Xtrain)[:, 0]
        _symbols = np.array(symbols_predict)[_companies]

        del model
        K.clear_session()

        forecast_indices = _forecast.argsort()[-num_stocks:]
        sorted_Xtrain = _Xtrain[forecast_indices,:,:,:]
        sorted_Ytrain = _Ytrain[forecast_indices]
        sorted_companies = _companies[forecast_indices]
        sorted_forecast = _forecast[forecast_indices]
        sorted_symbols = _symbols[forecast_indices]
        ##print("\n ... sorted_symbols = ",sorted_symbols[-num_stocks:])

#        ensemble_sharpe_weights = np.ones(np.array(sorted_symbols[-num_stocks:]).shape, 'float') * params['_sharpe_ratio_system']
#        ensemble_recent_sharpe_weights = np.ones_like(ensemble_sharpe_weights) * params['_sharpe_ratio_recent_system']
        ensemble_sharpe_weights = np.ones(sorted_companies.shape, 'float')
        ensemble_recent_sharpe_weights = np.ones_like(ensemble_sharpe_weights)
        #print("sorted_Xtrain.shape = ",sorted_Xtrain.shape, "   sorted_companies.shape = ", sorted_companies.shape)
        for icompany in range(sorted_companies.shape[0]):
            #print("sorted_Xtrain[icompany,:,2,0].shape, sharpe = ",sorted_Xtrain[icompany,:,2,0].shape,allstats((sorted_Xtrain[icompany,:,0,0]+1.).cumprod()).sharpe(periods_per_year=252./increments[2]))
            if sort_mode == 'sharpe':
                ensemble_sharpe_weights[icompany] = allstats((sorted_Xtrain[icompany,:,-1,0]+1.).cumprod()).sharpe(periods_per_year=252./increments[-1])
                ensemble_recent_sharpe_weights[icompany] = allstats((sorted_Xtrain[icompany,:,int(sorted_Xtrain.shape[2]/2),0]+1.).cumprod()).sharpe(periods_per_year=252./increments[0])
            elif sort_mode == 'sharpe_plus_sortino':
                ensemble_sharpe_weights[icompany] = allstats((sorted_Xtrain[icompany,:,-1,0]+1.).cumprod()).sharpe(periods_per_year=252./increments[-1]) + \
                                                    allstats((sorted_Xtrain[icompany,:,-1,0]+1.).cumprod()).sortino()
                ensemble_recent_sharpe_weights[icompany] = allstats((sorted_Xtrain[icompany,:,int(sorted_Xtrain.shape[2]/2),0]+1.).cumprod()).sharpe(periods_per_year=252./increments[0]) + \
                                                           allstats((sorted_Xtrain[icompany,:,int(sorted_Xtrain.shape[2]/2),0]+1.).cumprod()).sortino()
            elif sort_mode == 'sortino':
                ensemble_sharpe_weights[icompany] = allstats((sorted_Xtrain[icompany,:,-1,0]+1.).cumprod()).sortino()
                ensemble_recent_sharpe_weights[icompany] = allstats((sorted_Xtrain[icompany,:,int(sorted_Xtrain.shape[2]/2),0]+1.).cumprod()).sortino()
        ensemble_equal_weights = np.ones_like(ensemble_sharpe_weights)
        ensemble_rank_weights = np.arange(np.array(sorted_symbols[-num_stocks:]).shape[0])[::-1]

        ensemble_symbols.append(sorted_symbols[-num_stocks:])
        ensemble_Ytrain.append(sorted_Ytrain[-num_stocks:])
        ensemble_sharpe.append(ensemble_sharpe_weights)
        ensemble_recent_sharpe.append(ensemble_recent_sharpe_weights)
        ensemble_equal.append(ensemble_recent_sharpe_weights)
        ensemble_rank.append(ensemble_rank_weights)

        #print(imodel,sorted_symbols[-num_stocks:])
        #print(" ... ",ensemble_sharpe_weights)

    # sift through ensemble symbols
    ensemble_symbols = np.array(ensemble_symbols).flatten()
    ensemble_Ytrain = np.array(ensemble_Ytrain).flatten()
    ensemble_sharpe = np.array(ensemble_sharpe).flatten()
    ensemble_recent_sharpe = np.array(ensemble_recent_sharpe).flatten()
    ensemble_equal = np.array(ensemble_equal).flatten()
    ensemble_rank = np.array(ensemble_rank).flatten()

    #unique_symbols = list(set(np.array(ensemble_symbols)))
    unique_symbols = list(set(list(np.array(ensemble_symbols).flatten())))
    unique_ensemble_symbols = []
    unique_ensemble_Ytrain = []
    unique_ensemble_sharpe = []
    unique_ensemble_recent_sharpe = []
    unique_ensemble_equal = []
    unique_ensemble_rank = []
    for k, ksymbol in enumerate(unique_symbols):
        unique_ensemble_symbols.append(np.array(ensemble_symbols)[ensemble_symbols == ksymbol][0])
        unique_ensemble_Ytrain.append(ensemble_Ytrain[ensemble_symbols == ksymbol].mean())
        unique_ensemble_sharpe.append(ensemble_sharpe[ensemble_symbols == ksymbol].sum())
        unique_ensemble_recent_sharpe.append(ensemble_recent_sharpe[ensemble_symbols == ksymbol].sum())
        unique_ensemble_equal.append(ensemble_equal[ensemble_symbols == ksymbol].sum())
        unique_ensemble_rank.append(ensemble_rank[ensemble_symbols == ksymbol].sum())

    #print("unique_ensemble_sharpe = ", np.sort(unique_ensemble_sharpe)[-num_stocks:])

    indices_recent = np.argsort(unique_ensemble_recent_sharpe)[-num_stocks:]
    #print("indices = ",indices)
    sorted_recent_sharpe = np.array(unique_ensemble_recent_sharpe)[indices_recent]
    sorted_recent_sharpe = np.array(sorted_recent_sharpe)

    unique_ensemble_sharpe = np.array(unique_ensemble_sharpe) + np.array(unique_ensemble_recent_sharpe)

    indices = np.argsort(unique_ensemble_sharpe)[-num_stocks:]
    #print("indices = ",indices)
    sorted_sharpe = np.array(unique_ensemble_sharpe)[indices]
    sorted_sharpe = np.array(sorted_sharpe)
    #print("                                       ... sorted_sharpe[sorted_sharpe < 0.].shape = ", sorted_sharpe[sorted_sharpe < 0.].shape, sorted_recent_sharpe[sorted_recent_sharpe < 0.].shape)
    sorted_symbols = np.array(unique_ensemble_symbols)[indices]
    sorted_Ytrain = np.array(unique_ensemble_Ytrain)[indices]
    #company_indices = [list(unique_ensemble_symbols).index(isymbol) for isymbol in sorted_symbols]

    ##print("sorted_symbols = ", sorted_symbols)
    ##print("sorted_Ytrain = ", sorted_Ytrain)
    #print("_symbols[company_indices] = ", _symbols[company_indices][-num_stocks:])
    #print("_Ytrain[company_indices] = ", _Ytrain[company_indices][-num_stocks:])

    try:
        _Ytrain = _Ytrain[dates_predict == idate]
        sorted_Ytrain = sorted_Ytrain[-num_stocks:]
        BH_gain = _Ytrain.mean()
    except:
        BH_gain = 0.0

    avg_gain = sorted_Ytrain.mean()

    return avg_gain, BH_gain, sorted_symbols



def performance_metric(models_plotdates, avg_gain, sort_mode='sharpe', number_months=24):
    model_metric = []
    model_dates = []
    for ii in range(number_months,len(avg_gain)):
        #if models_plotdates[ii].month == 1:
        if models_plotdates[ii].month <= 12:
            performance_history = np.array(avg_gain[ii-number_months:ii])
            if sort_mode == 'sharpe':
                computed_stats = allstats(performance_history).sharpe(periods_per_year=12.)
                #print(ii,models_plotdates[ii],computed_stats)
            elif sort_mode == 'sortino':
                computed_stats = (allstats(performance_history).sortino())
            elif sort_mode == 'sharpe_plus_sortino':
                computed_stats = allstats(performance_history).sharpe(periods_per_year=12.) + allstats(performance_history).sortino()
            '''if computed_stats != empty_list:
                model_dates.append(models_plotdates[ii])
                model_metric.append(computed_stats)'''
            model_dates.append(models_plotdates[ii])
            if np.isinf(computed_stats):
                computed_stats = -1.
            model_metric.append(computed_stats)
    return model_dates, model_metric




# --------------------------------------------------
# Get program parameters.
# --------------------------------------------------

run_params = GetParams()


# --------------------------------------------------
# Import list of symbols to process.
# --------------------------------------------------

# read list of symbols from disk.
#stockList = 'Naz100'
stockList = run_params['stockList_predict']
if stockList == 'Naz100':
    filename = os.path.join(_data_path, 'symbols', 'Naz100_Symbols.txt')                   # plotmax = 1.e10, runnum = 902
elif stockList == 'SP500' or stockList == 'SP_wo_Naz':
    filename = os.path.join(_data_path, 'symbols', 'SP500_Symbols.txt')                   # plotmax = 1.e10, runnum = 902

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
if do_update is True:
    UpdateHDF5(symbols_directory, symbols_file)  ### assume hdf is already up to date
adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(filename)

firstdate = datearray[0]

# --------------------------------------------------
# Clean up missing values in input quotes
#  - infill interior NaN values using nearest good values to linearly interpolate
#  - copy first valid quote from valid date to all earlier positions
#  - copy last valid quote from valid date to all later positions
# --------------------------------------------------

for ii in range(adjClose.shape[0]):
    adjClose[ii, :] = interpolate(adjClose[ii, :])
    adjClose[ii, :] = cleantobeginning(adjClose[ii, :])
    adjClose[ii, :] = cleantoend(adjClose[ii, :])

print(" security values check: ", adjClose[np.isnan(adjClose)].shape)

# --------------------------------------------------
# prepare labeled data for DL training
# - set up historical data plus actual performance one month forward
# --------------------------------------------------

best_final_value = -99999
best_recent_final_value = -99999
#num_periods_history = 20
#first_history_index = 1500
num_periods_history = run_params['num_periods_history']
first_history_index = run_params['first_history_index']

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
    if ii is True:
        datearray_new_months.append(dates[i])

datearray_new_months = list(set(datearray_new_months))
datearray_new_months.sort()


# --------------------------------------------------
# make predictions monthly for backtesting
# - apply multiple DL models and use 'num_stocks' most frequent stocks
# - break ties randomly
# --------------------------------------------------

#for model_filter in ['SP', 'Naz100', 'all']:
#model_filter = 'SP'
model_filter = run_params['model_filter']

models_folder = os.path.join(os.getcwd(), 'pngs')
models_list = os.listdir(models_folder)
models_list = [i for i in models_list if '.txt' in i]
models_list = [i for i in models_list if 'bak' not in i]

if model_filter == 'Naz100':
    models_list = [i for i in models_list if 'Naz100' in i]
if model_filter == 'SP':
    models_list = [i for i in models_list if 'SP' in i]

#sort_mode = 'sharpe'
#sort_mode = 'sortino'
#sort_mode = 'sharpe_plus_sortino'
sort_mode = run_params['sort_mode']

print("\n\n****************************\n")
print(" ... model_filter = ", model_filter)
print(" ... sort_mode = ", sort_mode)

#inum_stocks = 7
inum_stocks = run_params['num_stocks']
print(" ... inum_stocks = ", inum_stocks)

months_for_performance_comparison = run_params['months_for_performance_comparison']

# create folder for best performing output, if it doesn't exist
if not os.path.exists(os.path.join(models_folder, run_params['folder_for_best_performers'])):
        os.mkdir(os.path.join(models_folder, run_params['folder_for_best_performers']))

for i_num_months in [months_for_performance_comparison]:

    print("\n\n\n******************************************************************")
    print(" number months for sharpe or sortino = ", i_num_months)
    print("******************************************************************\n")

    plt.clf()
    all_model_dates = []
    all_model_metrics_sharpe = []
    all_model_metrics_sortino = []
    sortino_list = []
    sharpe_list = []
    all_model_names = []
    cumu_models = []
    system_final_values = []
    print(" ... beginning...")

    for im, imodel in enumerate(models_list):

        try:
            avg_gain, _, _, models_plotdates = one_model_prediction(os.path.abspath(os.path.join(models_folder,imodel)),
                                                     first_history_index,
                                                     datearray,
                                                     adjClose,
                                                     symbols,
                                                     inum_stocks)
            plt.plot(models_plotdates, avg_gain,label=str(im))
            cumu_models.append(avg_gain)

            system_final_values.append(avg_gain[-1])

            print(" ... "+str(im)+" of "+str(len(models_list)))
            model_sharpe = allstats(np.array(avg_gain)).sharpe(periods_per_year=12.)
            model_sortino = allstats(np.array(avg_gain)).sortino()
            sharpe_list.append(model_sharpe)
            sortino_list.append(model_sortino)
            print(" ... sharpe = ", model_sharpe)
            print(" ... sortino = ", model_sortino)

            model_dates, model_metric_sharpe = performance_metric(models_plotdates, avg_gain, sort_mode='sharpe', number_months=i_num_months)
            model_dates, model_metric_sortino = performance_metric(models_plotdates, avg_gain, sort_mode='sortino', number_months=i_num_months)
            print(" ... sharpe and sortino ranks are the same ",(np.argsort(model_metric_sharpe)==np.argsort(model_metric_sortino)).all() )
            print(" ... sharpe min/mean/median/max = ", np.min(model_metric_sharpe), np.mean(model_metric_sharpe), np.median(model_metric_sharpe), np.max(model_metric_sharpe))
            print(" ... sortino min/mean/median/max = ", np.min(model_metric_sortino), np.mean(model_metric_sortino), np.median(model_metric_sortino), np.max(model_metric_sortino))
            all_model_dates.append(model_dates)
            all_model_metrics_sharpe.append(model_metric_sharpe)
            all_model_metrics_sortino.append(model_metric_sortino)
            all_model_names.append(imodel)
        except:
            #print(" ... error line 695 ...")
            continue

    plt.grid(True)
    plt.yscale('log')
    plt.legend()


    sharpe_list = np.array(sharpe_list)
    sortino_list= np.array(sortino_list)

    sharpe_shortlist = []
    sortino_shortlist = []
    for i, imodel in enumerate(all_model_names):
        if system_final_values[i] > run_params['final_system_value_threshold']:
            sharpe_shortlist.append(sharpe_list[i])
            sortino_shortlist.append(sortino_list[i])
   # sharpe_threshold = np.median(sharpe_shortlist)
   # sortino_threshold = np.median(sortino_shortlist)
    sharpe_shortlist = np.array(sharpe_shortlist)
    sortino_shortlist = np.array(sortino_shortlist)
    sharpe_threshold = np.percentile(sharpe_shortlist, run_params['sharpe_threshold_percentile'])
    sortino_threshold = np.percentile(sortino_shortlist, run_params['sortino_threshold_percentile'])

    useless_models = []
    model_count_sharpe = []
    model_count_sortino = []
    for i, imodel in enumerate(all_model_names):
        aaa = np.array(all_model_metrics_sharpe).argmax(axis=0)
        model_count_sharpe.append( aaa[aaa == i].shape[0] )
        aaa = np.array(all_model_metrics_sortino).argmax(axis=0)
        model_count_sortino.append( aaa[aaa == i].shape[0] )
        print(" model, sharpe count, sortino count = ",
              (imodel+" "*25)[:35],
              format(system_final_values[i],'12,.0f'),
              format(model_count_sharpe[-1],'>4d'),
              format(model_count_sortino[-1],'>4d'))
        import shutil
        '''
        if model_count_sharpe[-1] > 0 and system_final_values[i] > 10000000.:
            shutil.copy2(imodel, 'best_sharpe2')
            shutil.copy2(imodel.replace('txt', 'hdf'), 'best_sharpe2')
            shutil.copy2(imodel.replace('txt', 'json'), 'best_sharpe2')
            shutil.copy2(imodel.replace('.txt', '_fig-2.png'), 'best_sharpe2')
            shutil.copy2(imodel.replace('.txt', '_fig-3.png'), 'best_sharpe2')
            shutil.copy2(imodel.replace('.txt', '_fig-4.png'), 'best_sharpe2')
            shutil.copy2(imodel.replace('.txt', '_crossplot.png'), 'best_sharpe2')
        if model_count_sortino[-1] > 0  and system_final_values[i] > 10000000.:
            shutil.copy2(imodel, 'best_sortino2')
            shutil.copy2(imodel.replace('txt', 'hdf'), 'best_sortino2')
            shutil.copy2(imodel.replace('txt', 'json'), 'best_sortino2')
            shutil.copy2(imodel.replace('.txt', '_fig-2.png'), 'best_sortino2')
            shutil.copy2(imodel.replace('.txt', '_fig-3.png'), 'best_sortino2')
            shutil.copy2(imodel.replace('.txt', '_fig-4.png'), 'best_sortino2')
            shutil.copy2(imodel.replace('.txt', '_crossplot.png'), 'best_sortino2')
        if (model_count_sharpe[-1] == 0 and model_count_sortino[-1] == 0) or system_final_values[i] < 10000000.:
            useless_models.append(imodel)
        '''
        #if system_final_values[i] > run_params['final_system_value_threshold'].:
        if system_final_values[i] > run_params['final_system_value_threshold'] and (sharpe_list[i] > sharpe_threshold or sortino_list[i] > sortino_threshold):
            shutil.copy2(imodel, run_params['folder_for_best_performers'])
            shutil.copy2(imodel.replace('txt', 'hdf'), run_params['folder_for_best_performers'])
            shutil.copy2(imodel.replace('txt', 'json'), run_params['folder_for_best_performers'])
            shutil.copy2(imodel.replace('.txt', '_fig-2.png'), run_params['folder_for_best_performers'])
            shutil.copy2(imodel.replace('.txt', '_fig-3.png'), run_params['folder_for_best_performers'])
            shutil.copy2(imodel.replace('.txt', '_fig-4.png'), run_params['folder_for_best_performers'])
            shutil.copy2(imodel.replace('.txt', '_crossplot.png'), run_params['folder_for_best_performers'])
        else:
            useless_models.append(imodel)


    print("\n ... count of sharpe indices = ", model_count_sharpe)
    print(" ... count of best sortino indices = ", model_count_sortino)

    best_sharpe_indices = list(set(np.array(all_model_metrics_sharpe).argmax(axis=0)))
    best_sortino_indices = list(set(np.array(all_model_metrics_sortino).argmax(axis=0)))

    print("\n ... best sharpe indices = ", best_sharpe_indices)
    print(" ... best sortino indices = ", best_sortino_indices)



