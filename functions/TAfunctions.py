import os
import sys
import datetime
import numpy as np
from numpy import isnan
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

from functions.allstats import *
from matplotlib.pylab import *

#---------------------------------------------

def _is_odd(number):
    if int(number)%2 == 0:
        return False
    else:
        if number == int(number):
            return True
        else:
            return False

def _substringfinder(mylist, pattern):
    # count and return the number of matches pattern has in mylist
    # example:
    #   _substringfinder ([1.0,1.0,5.,.6,4.,3.,1.,1.000],[1.,1.]) returns 2
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(pattern)
    return len(matches)


def fix_params_file(config_filename, verbose=False):
    import os

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

    try:
        use_leaky_relu = parser.get("training_params", "use_leaky_relu")
        leaky_relu_alpha = parser.get("training_params", "leaky_relu_alpha")
    except:
        use_leaky_relu = False
        leaky_relu_alpha = 0.05

    try:
        use_separable = parser.get("training_params", "use_separable")
    except:
        use_separable = False

    num_stocks = parser.get("training_params", "num_stocks")
    increments = parser.get("training_params", "increments")
    num_periods_history = parser.get("training_params", "num_periods_history")
    first_history_index = parser.get("training_params", "first_history_index")
    try:
        _sharpe_ratio_system = parser.get("training_params", "_sharpe_ratio_system")
    except:
        _sharpe_ratio_system = 0.
    try:
        _sharpe_ratio_recent_system = parser.get("training_params", "_sharpe_ratio_recent_system")
    except:
        _sharpe_ratio_recent_system = 0.

    weights_filename = parser.get("training_params", "weights_filename")
    model_json_filename = parser.get("training_params", "model_json_filename")

    # put params in a dictionary
    params = {}
    params['perform_batch_normalization'] = perform_batch_normalization
    params['use_dense_layers'] = use_dense_layers
    params['use_leaky_relu'] = use_leaky_relu
    params['use_separable'] = use_separable
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
    #print("params = ", params)

    return params


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
        if idate != dates_predict[-1] and companies_predict[i] < companies_predict[i-1]:
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


def ensemble_prediction(models_folder, models_list, idate, datearray, adjClose, symbols, num_stocks, sort_mode='sharpe', verbose=False):

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
            elif sort_mode == 'count' or sort_mode == 'equal':
                ensemble_sharpe_weights[icompany] = 1.
                ensemble_recent_sharpe_weights[icompany] = 1.

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
    if sort_mode == 'equal':
        unique_ensemble_sharpe = np.ones_like(unique_ensemble_sharpe)

    indices = np.argsort(unique_ensemble_sharpe)[-num_stocks:]
    #print("indices = ",indices)
    sorted_sharpe = np.array(unique_ensemble_sharpe)[indices]
    ##sorted_sharpe = np.array(sorted_sharpe)
    #print("                                       ... sorted_sharpe[sorted_sharpe < 0.].shape = ", sorted_sharpe[sorted_sharpe < 0.].shape, sorted_recent_sharpe[sorted_recent_sharpe < 0.].shape)
    sorted_symbols = np.array(unique_ensemble_symbols)[indices]
    sorted_Ytrain = np.array(unique_ensemble_Ytrain)[indices]
    #company_indices = [list(unique_ensemble_symbols).index(isymbol) for isymbol in sorted_symbols]

    ##print("sorted_symbols = ", sorted_symbols)
    ##print("sorted_Ytrain = ", sorted_Ytrain)
    #print("_symbols[company_indices] = ", _symbols[company_indices][-num_stocks:])
    #print("_Ytrain[company_indices] = ", _Ytrain[company_indices][-num_stocks:])

    """
    # equal-weighted
    try:
        _Ytrain = _Ytrain[dates_predict == idate]
        sorted_Ytrain = sorted_Ytrain[-num_stocks:]
        BH_gain = _Ytrain.mean()
    except:
        BH_gain = 0.0

    avg_gain = sorted_Ytrain.mean()

    return avg_gain, BH_gain, sorted_symbols
    """

    # weighted contribution according to sharpe ratio
    try:
        _Ytrain = _Ytrain[dates_predict == idate]
        sorted_Ytrain = sorted_Ytrain[-num_stocks:]
        sorted_sharpe = sorted_sharpe[-num_stocks:]
        BH_gain = _Ytrain.mean()
    except:
        BH_gain = 0.0

    ####sorted_sharpe = 1./ sorted_sharpe
    sorted_sharpe = np.sqrt(np.clip(sorted_sharpe,0.,sorted_sharpe.max()))
    if verbose:
        print("       gains, stddev of gains = ", np.around(sorted_Ytrain,3),np.std(sorted_Ytrain))
        print("       sorted_sharpe", np.around(sorted_sharpe,2), np.std(sorted_sharpe))
        print("       weights", np.around(sorted_sharpe/ sorted_sharpe.sum(),3), np.std(sorted_sharpe/ sorted_sharpe.sum()))
        print("       weighted gains", np.around((sorted_Ytrain * sorted_sharpe) / sorted_sharpe.sum(),3))
    avg_gain = ((sorted_Ytrain * sorted_sharpe) / sorted_sharpe.sum()).sum()
    symbols_weights = sorted_sharpe / sorted_sharpe.sum()

    return avg_gain, BH_gain, sorted_symbols, symbols_weights



def ensemble_stock_choice(models_folder, models_list, idate, datearray, adjClose, symbols, num_stocks, sort_mode='sharpe', verbose=False):

    #--------------------------------------------------------------
    # loop through best models and pick companies from ensemble prediction
    #--------------------------------------------------------------

    ensemble_symbols = []
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
        first_history_index = params['first_history_index']

        symbols_predict = symbols
        '''
        print("     ..... idate = ", idate)
        print("     ..... datearray = ", datearray[0], datearray[-1])
        print("     ..... adjClose = ", adjClose.shape)
        print("     ..... first_history_index = ", first_history_index)
        print("     ..... num_periods_history = ", num_periods_history)
        print("     ..... increments = ", increments)
        '''
        Xpredict, dates_predict, companies_predict = generatePredictionInput3layer(idate,
                                                                                   datearray,
                                                                                   adjClose,
                                                                                   first_history_index,
                                                                                   num_periods_history,
                                                                                   increments,
                                                                                   output_incr='monthly',
                                                                                   verbose=False)

        dates_predict = np.array(dates_predict)
        companies_predict = np.array(companies_predict)

        # --------------------------------------------------
        # make predictions monthly for backtesting
        # - there might be some bias since entire period
        #   has data used for training
        # --------------------------------------------------

        weights_filename = os.path.join(models_folder, imodel)
        try:
            model.load_weights(weights_filename)
        except:
            pass

        # show predictions for (single) last date
        _Xtrain = Xpredict[dates_predict == idate]
        #_dates = np.array(dates_predict[dates_predict == idate])
        _companies = np.array(companies_predict[dates_predict == idate])
        _forecast = model.predict(_Xtrain)[:, 0]
        _symbols = np.array(symbols_predict)[_companies]

        del model
        K.clear_session()

        forecast_indices = _forecast.argsort()[-num_stocks:]
        sorted_Xtrain = _Xtrain[forecast_indices,:,:,:]
        sorted_companies = _companies[forecast_indices]
        sorted_symbols = _symbols[forecast_indices]

        ensemble_sharpe_weights = np.ones(sorted_companies.shape, 'float')
        ensemble_recent_sharpe_weights = np.ones_like(ensemble_sharpe_weights)
        for icompany in range(sorted_companies.shape[0]):
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
            elif sort_mode == 'count' or sort_mode == 'equal':
                ensemble_sharpe_weights[icompany] = 1.
                ensemble_recent_sharpe_weights[icompany] = 1.

        ensemble_rank_weights = np.arange(np.array(sorted_symbols[-num_stocks:]).shape[0])[::-1]

        ensemble_symbols.append(sorted_symbols[-num_stocks:])
        ensemble_sharpe.append(ensemble_sharpe_weights)
        ensemble_recent_sharpe.append(ensemble_recent_sharpe_weights)
        ensemble_equal.append(ensemble_recent_sharpe_weights)
        ensemble_rank.append(ensemble_rank_weights)

    # sift through ensemble symbols
    ensemble_symbols = np.array(ensemble_symbols).flatten()
    ensemble_sharpe = np.array(ensemble_sharpe).flatten()
    ensemble_recent_sharpe = np.array(ensemble_recent_sharpe).flatten()
    ensemble_equal = np.array(ensemble_equal).flatten()
    ensemble_rank = np.array(ensemble_rank).flatten()

    unique_symbols = list(set(list(np.array(ensemble_symbols).flatten())))
    unique_ensemble_symbols = []
    unique_ensemble_sharpe = []
    unique_ensemble_recent_sharpe = []
    unique_ensemble_equal = []
    unique_ensemble_rank = []
    for k, ksymbol in enumerate(unique_symbols):
        unique_ensemble_symbols.append(np.array(ensemble_symbols)[ensemble_symbols == ksymbol][0])
        unique_ensemble_sharpe.append(ensemble_sharpe[ensemble_symbols == ksymbol].sum())
        unique_ensemble_recent_sharpe.append(ensemble_recent_sharpe[ensemble_symbols == ksymbol].sum())
        unique_ensemble_equal.append(ensemble_equal[ensemble_symbols == ksymbol].sum())
        unique_ensemble_rank.append(ensemble_rank[ensemble_symbols == ksymbol].sum())

    indices_recent = np.argsort(unique_ensemble_recent_sharpe)[-num_stocks:]
    sorted_recent_sharpe = np.array(unique_ensemble_recent_sharpe)[indices_recent]
    sorted_recent_sharpe = np.array(sorted_recent_sharpe)

    unique_ensemble_sharpe = np.array(unique_ensemble_sharpe) + np.array(unique_ensemble_recent_sharpe)
    if sort_mode == 'equal':
        unique_ensemble_sharpe = np.ones_like(unique_ensemble_sharpe)

    indices = np.argsort(unique_ensemble_sharpe)[-num_stocks:]
    sorted_sharpe = np.array(unique_ensemble_sharpe)[indices]
    sorted_symbols = np.array(unique_ensemble_symbols)[indices]

    # weighted contribution according to sharpe ratio
    sorted_sharpe = sorted_sharpe[-num_stocks:]

    sorted_sharpe = np.sqrt(np.clip(sorted_sharpe,0.,sorted_sharpe.max()))
    if verbose:
        print("       sorted_sharpe", np.around(sorted_sharpe,2), np.std(sorted_sharpe))
        print("       weights", np.around(sorted_sharpe/ sorted_sharpe.sum(),3), np.std(sorted_sharpe/ sorted_sharpe.sum()))
    symbols_weights = sorted_sharpe / sorted_sharpe.sum()

    return sorted_symbols, symbols_weights



"""
def generateExamples(datearray,adjClose,first_history_index,num_periods_history,increments,output_incr='monthly'):

    # output_incr can be 'monthly', 'weekly', or 'daily'

    shortest_incr = increments[0]
    middle_incr = increments[1]
    long_incr = increments[2]

    invalid_count = 0
    dates = []
    rgb_image = np.array([])

    first_index = max( int(first_history_index), int(long_incr*num_periods_history) )

    for jdate in range(first_index,adjClose.shape[1]-22):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day
        if output_incr == 'monthly':
            output_test = jdate == first_history_index or datearray[jdate].year > datearray[jdate-1].year or datearray[jdate].month > datearray[jdate-1].month
        if output_incr == 'midmonth':
            output_test = jdate == first_history_index \
            or datearray[jdate].weekday() < datearray[jdate-1].weekday() \
            and (12 <= datearray[jdate].day < 19)
        elif output_incr == 'weekly':
            output_test = jdate == first_history_index or datearray[jdate].weekday() < datearray[jdate-1].weekday()
        elif output_incr == 'daily':
            output_test = True

        if datearray[jdate].year > datearray[jdate-1].year:
            print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

        if output_test == True:

            # create a row for each company
            num_companies = 0
            avg_forallcompanies = np.zeros((num_periods_history,3),'float')
            for iCompany in range(adjClose.shape[0]):

                r_array = adjClose[iCompany,jdate-num_periods_history*shortest_incr:jdate+shortest_incr:shortest_incr].copy()
                g_array = adjClose[iCompany,jdate-num_periods_history*middle_incr:jdate+middle_incr:middle_incr].copy()
                b_array = adjClose[iCompany,jdate-num_periods_history*long_incr:jdate+long_incr:long_incr].copy()

                if r_array[0]==0. or g_array[0]==0. or b_array[0]==0.:
                    invalid_count += 1
                    continue

                # filter out series without valid quotes at beginning
                r_list = list(r_array/r_array[0])
                g_list = list(g_array/g_array[0])
                b_list = list(b_array/b_array[0])

                r_count_invalid = _substringfinder(r_list, [1.0,1.0])
                g_count_invalid = _substringfinder(g_list, [1.0,1.0])
                b_count_invalid = _substringfinder(b_list, [1.0,1.0])

                if r_count_invalid>0 or g_count_invalid>0 or b_count_invalid>0:
                    invalid_count += 1
                    continue

                if r_array[:-1].min()==0. or g_array[:-1].min()==0. or b_array[:-1].min()==0.:
                    invalid_count += 1
                    continue

                # save as gain/loss for period (drop first value)
                r_array = r_array[1:] / r_array[:-1] - 1.
                g_array = g_array[1:] / g_array[:-1] - 1.
                b_array = b_array[1:] / b_array[:-1] - 1.

                # combine and reshpae
                rgb_row = np.vstack((r_array,g_array,b_array,r_array,g_array,b_array))
                rgb_row = np.swapaxes(rgb_row,0,1)
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],6,1))

                try:
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    forecast = np.hstack((forecast,adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    forecast = adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                avg_forallcompanies += rgb_row[:,3:,0]

            avg_forallcompanies /= num_companies
            for iCompany in range(1,num_companies+1):
                rgb_image[:,3:,-iCompany] -= avg_forallcompanies

    rgb_image = rgb_image.swapaxes(0,2)
    rgb_image = rgb_image.swapaxes(1,2)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],20,3,2))
    forecast = forecast.reshape(forecast.shape[0],1)

    print(" ... Labeled data successfully created ...")
    return rgb_image, forecast, dates, company_name
"""


#---------------------------------------------

def generateExamples(datearray,adjClose,first_history_index,num_periods_history,increments,output_incr='monthly'):

    # output_incr can be 'monthly', 'weekly', or 'daily'

    # make certain 'increments' is sorted in ascending order
    increments.sort()

    invalid_count = 0
    dates = []
    rgb_image = np.array([])

    first_index = max( int(first_history_index), int(increments[-1]*num_periods_history) )

    for jdate in range(first_index,adjClose.shape[1]-22):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day
        if output_incr == 'monthly':
            output_test = jdate == first_history_index or datearray[jdate].year > datearray[jdate-1].year or datearray[jdate].month > datearray[jdate-1].month
        if output_incr == 'midmonth':
            output_test = jdate == first_history_index \
            or datearray[jdate].weekday() < datearray[jdate-1].weekday() \
            and (12 <= datearray[jdate].day < 19)
        elif output_incr == 'weekly':
            output_test = jdate == first_history_index or datearray[jdate].weekday() < datearray[jdate-1].weekday()
        elif output_incr == 'daily':
            output_test = True

        if datearray[jdate].year > datearray[jdate-1].year:
            print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

        if output_test == True:

            # create a row for each company
            num_companies = 0
            avg_forallcompanies = np.zeros((num_periods_history,len(increments)),'float')

            for iCompany in range(adjClose.shape[0]):

                _quotes_image = np.zeros((num_periods_history+1,len(increments)),'float')
                for i,iincr in enumerate(increments):

                    _quotes_image[:,i] = adjClose[iCompany,jdate-num_periods_history*iincr:jdate+1:iincr].copy()

                    if iincr==increments[0]:
                        invalid_quote_count = 0
                    if _quotes_image[0,i]==0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    if _quotes_image[-1,i] == 0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    nantest = _quotes_image[:,i]
                    if nantest[np.isnan(nantest)].shape[0] > 0 or nantest[np.isinf(nantest)].shape[0] > 0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    print("invalid quotes for iCompany,date : ",iCompany, datearray[jdate])
                    continue

                # save as gain/loss for period (drop first value)
                _quotes_image = _quotes_image[1:,:] / _quotes_image[:-1,:] - 1.

                for i,iincr in enumerate(increments):

                    if iincr==increments[0]:
                        invalid_quote_count = 0

                    _quote_image_list = list(_quotes_image[:,i])
                    _quote_image_list_invalid = _substringfinder(_quote_image_list, [0.0,0.0])

                    if _quote_image_list_invalid>0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    continue

                # combine and reshape
                #rgb_row = np.dstack((_quotes_image,_quotes_image)).swapaxes(-2,-1)
                rgb_row = np.hstack((_quotes_image,_quotes_image))
                #print ("rgb_row.shape = ", rgb_row.shape)
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],2*len(increments),1))

                try:
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    forecast = np.hstack((forecast,adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    forecast = adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                avg_forallcompanies += rgb_row[:,len(increments):,0]

            avg_forallcompanies /= num_companies
            for iCompany in range(1,num_companies+1):
                rgb_image[:,len(increments):,-iCompany] -= avg_forallcompanies

    #print ("rgb_image.shape = ", rgb_image.shape)
    rgb_image = rgb_image.swapaxes(0,2)
    rgb_image = rgb_image.swapaxes(1,2)
    #print ("rgb_image.shape = ", rgb_image.shape)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],num_periods_history,2,len(increments)))
    forecast = forecast.reshape(forecast.shape[0],1)
    rgb_image = rgb_image.swapaxes(-2,-1)

    print(" ... Labeled data successfully created ...")
    return rgb_image, forecast, dates, company_name


def generateExamplesForDate(predict_date,datearray,adjClose,num_periods_history,increments,output_incr='monthly',verbose=False):

    # output_incr can be 'monthly', 'weekly', or 'daily'

    # make certain 'increments' is sorted in ascending order
    increments.sort()

    invalid_count = 0
    dates = []
    rgb_image = np.array([])

    first_index = np.argmin(np.abs(np.array(datearray)-predict_date))

    for jdate in range(first_index, first_index +1):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day
        if output_incr == 'monthly':
            output_test = jdate == first_history_index or datearray[jdate].year > datearray[jdate-1].year or datearray[jdate].month > datearray[jdate-1].month
        if output_incr == 'midmonth':
            output_test = jdate == first_history_index \
            or datearray[jdate].weekday() < datearray[jdate-1].weekday() \
            and (12 <= datearray[jdate].day < 19)
        elif output_incr == 'weekly':
            output_test = jdate == first_history_index or datearray[jdate].weekday() < datearray[jdate-1].weekday()
        elif output_incr == 'daily':
            output_test = True

        if datearray[jdate].year > datearray[jdate-1].year:
            if verbose:
                print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

        if output_test == True:

            # create a row for each company
            num_companies = 0
            avg_forallcompanies = np.zeros((num_periods_history,len(increments)),'float')

            for iCompany in range(adjClose.shape[0]):

                _quotes_image = np.zeros((num_periods_history+1,len(increments)),'float')
                for i,iincr in enumerate(increments):

                    _quotes_image[:,i] = adjClose[iCompany,jdate-num_periods_history*iincr:jdate+1:iincr].copy()

                    if iincr==increments[0]:
                        invalid_quote_count = 0
                    if _quotes_image[0,i]==0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    if _quotes_image[-1,i] == 0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    nantest = _quotes_image[:,i]
                    if nantest[np.isnan(nantest)].shape[0] > 0 or nantest[np.isinf(nantest)].shape[0] > 0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    if verbose:
                        print("invalid quotes for iCompany,date : ",iCompany, datearray[jdate])
                    continue

                # save as gain/loss for period (drop first value)
                _quotes_image = _quotes_image[1:,:] / _quotes_image[:-1,:] - 1.

                for i,iincr in enumerate(increments):

                    if iincr==increments[0]:
                        invalid_quote_count = 0

                    _quote_image_list = list(_quotes_image[:,i])
                    _quote_image_list_invalid = _substringfinder(_quote_image_list, [0.0,0.0])

                    if _quote_image_list_invalid>0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    continue

                # combine and reshape
                #rgb_row = np.dstack((_quotes_image,_quotes_image)).swapaxes(-2,-1)
                rgb_row = np.hstack((_quotes_image,_quotes_image))
                #print ("rgb_row.shape = ", rgb_row.shape)
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],2*len(increments),1))

                try:
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    forecast = np.hstack((forecast,adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    forecast = adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                avg_forallcompanies += rgb_row[:,len(increments):,0]

            avg_forallcompanies /= num_companies
            for iCompany in range(1,num_companies+1):
                rgb_image[:,len(increments):,-iCompany] -= avg_forallcompanies

    #print ("rgb_image.shape = ", rgb_image.shape)
    rgb_image = rgb_image.swapaxes(0,2)
    rgb_image = rgb_image.swapaxes(1,2)
    #print ("rgb_image.shape = ", rgb_image.shape)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],num_periods_history,2,len(increments)))
    forecast = forecast.reshape(forecast.shape[0],1)
    rgb_image = rgb_image.swapaxes(-2,-1)

    if verbose:
        print(" ... Labeled data successfully created ...")
    return rgb_image, forecast, dates, company_name


def generatePredictionInput(predict_date,datearray,adjClose,first_history_index,num_periods_history,increments,output_incr='monthly',verbose=False):

    # predict_date is a single date for which input will be re-formatted to make predictions using DL

    # make certain 'increments' is sorted in ascending order
    increments.sort()

    invalid_count = 0
    dates = []
    rgb_image = np.array([])


    for jdate in range(adjClose.shape[1]):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day

        output_test = False

        if datearray[jdate] == predict_date:
            output_test = True

        if datearray[jdate].year > datearray[jdate-1].year and verbose==True:
            print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

        if output_test == True:

            # create a row for each company
            num_companies = 0
            avg_forallcompanies = np.zeros((num_periods_history,len(increments)),'float')

            for iCompany in range(adjClose.shape[0]):

                _quotes_image = np.zeros((num_periods_history+1,len(increments)),'float')
                for i,iincr in enumerate(increments):

                    _quotes_image[:,i] = adjClose[iCompany,jdate-num_periods_history*iincr:jdate+1:iincr].copy()

                    if iincr==increments[0]:
                        invalid_quote_count = 0
                    if _quotes_image[0,i]==0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    if _quotes_image[-1,i] == 0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    nantest = _quotes_image[:,i]
                    if nantest[np.isnan(nantest)].shape[0] > 0 or nantest[np.isinf(nantest)].shape[0] > 0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    print("invalid quotes for iCompany,date : ",iCompany, datearray[jdate])
                    continue

                # save as gain/loss for period (drop first value)
                _quotes_image = _quotes_image[1:,:] / _quotes_image[:-1,:] - 1.

                for i,iincr in enumerate(increments):

                    if iincr==increments[0]:
                        invalid_quote_count = 0

                    _quote_image_list = list(_quotes_image[:,i])
                    _quote_image_list_invalid = _substringfinder(_quote_image_list, [0.0,0.0])

                    if _quote_image_list_invalid>0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    continue

                #print("_quotes_image[:,0] = ",_quotes_image[:,0])
                #print("_quotes_image[:,1] = ",_quotes_image[:,1])

                # combine and reshape
                rgb_row = np.hstack((_quotes_image,_quotes_image))
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],2*len(increments),1))

                try:
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                avg_forallcompanies += rgb_row[:,len(increments):,0]

            avg_forallcompanies /= num_companies
            for iCompany in range(1,num_companies+1):
                rgb_image[:,len(increments):,-iCompany] -= avg_forallcompanies

    rgb_image = rgb_image.swapaxes(0,2)
    rgb_image = rgb_image.swapaxes(1,2)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],num_periods_history,2,len(increments)))
    rgb_image = rgb_image.swapaxes(-2,-1)

    print(" ... Labeled data successfully created ...")
    return rgb_image, dates, company_name


def generateExamples3layer(datearray,adjClose,first_history_index,num_periods_history,increments,output_incr='monthly',verbose=False):

    # output_incr can be 'monthly', 'weekly', or 'daily'

    # make certain 'increments' is sorted in ascending order
    increments.sort()

    invalid_count = 0
    dates = []
    rgb_image = np.array([])

    first_index = max( int(first_history_index), int(increments[-1]*num_periods_history) )

    for jdate in range(first_index,adjClose.shape[1]-22):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day
        if output_incr == 'monthly':
            output_test = jdate == first_history_index or datearray[jdate].year > datearray[jdate-1].year or datearray[jdate].month > datearray[jdate-1].month
        if output_incr == 'midmonth':
            output_test = jdate == first_history_index \
            or datearray[jdate].weekday() < datearray[jdate-1].weekday() \
            and (12 <= datearray[jdate].day < 19)
        elif output_incr == 'weekly':
            output_test = jdate == first_history_index or datearray[jdate].weekday() < datearray[jdate-1].weekday()
        elif output_incr == 'daily':
            output_test = True

        if datearray[jdate].year > datearray[jdate-1].year:
            if verbose:
                print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

        if output_test == True:

            # create a row for each company
            num_companies = 0
            summ = np.zeros((num_periods_history,len(increments)),'float')
            sumSquared = np.zeros((num_periods_history,len(increments)),'float')

            for iCompany in range(adjClose.shape[0]):

                _quotes_image = np.zeros((num_periods_history+1,len(increments)),'float')
                for i,iincr in enumerate(increments):

                    _quotes_image[:,i] = adjClose[iCompany,jdate-num_periods_history*iincr:jdate+1:iincr].copy()

                    if iincr==increments[0]:
                        invalid_quote_count = 0
                    if _quotes_image[0,i]==0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    if _quotes_image[-1,i] == 0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    nantest = _quotes_image[:,i]
                    if nantest[np.isnan(nantest)].shape[0] > 0 or nantest[np.isinf(nantest)].shape[0] > 0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    if verbose:
                        print("invalid quotes for iCompany,date : ",iCompany, datearray[jdate])
                    continue

                # save as gain/loss for period (drop first value)
                _quotes_image = _quotes_image[1:,:] / _quotes_image[:-1,:] - 1.

                for i,iincr in enumerate(increments):

                    if iincr==increments[0]:
                        invalid_quote_count = 0

                    _quote_image_list = list(_quotes_image[:,i])
                    _quote_image_list_invalid = _substringfinder(_quote_image_list, [0.0,0.0])

                    if _quote_image_list_invalid>0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    continue

                #print("_quotes_image[:,0] = ",_quotes_image[:,0])
                #print("_quotes_image[:,1] = ",_quotes_image[:,1])
                #print("_quotes_image[:,2] = ",_quotes_image[:,2])

                # combine and reshape
                rgb_row = np.dstack((_quotes_image,_quotes_image,_quotes_image)).swapaxes(1,2)
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],3*len(increments),1))
                #print('rgb_row.shape = ', rgb_row.shape)
                #print('rgb_row = ', rgb_row)

                try:
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    forecast = np.hstack((forecast,adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    forecast = adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                summ += rgb_row[:,:len(increments),0]
                sumSquared += rgb_row[:,:len(increments),0]**2
                #print (" ... rgb_row.shape = ",rgb_row.shape,rgb_row)

            #print (" ... summ = ",summ)
            #print (" ... sumSquared = ",sumSquared)
            avg_forallcompanies = summ / num_companies
            var_forallcompanies = (sumSquared - (summ**2)/num_companies)/(num_companies-1)
            #print (" ... avg_forallcompanies = ",avg_forallcompanies)
            #print (" ... var_forallcompanies = ",var_forallcompanies)
            #print (" ... rgb_image.shape = ",rgb_image.shape)
            #print (" ... rgb_row.shape = ",rgb_row.shape)

            for iCompany in range(1,num_companies+1):
                rgb_image[:,len(increments):2*len(increments),-iCompany] -= avg_forallcompanies
                rgb_image[:,2*len(increments):,-iCompany] = var_forallcompanies
            #print('gain/loss rgb_image[:,:3,-1] = ',rgb_image[:,:3,-1])
            #print('wo mean   rgb_image[:,3:6,-1] = ',rgb_image[:,3:6,-1])
            #print('variance  rgb_image[:,6:9,-1] = ',rgb_image[:,6:9,-1])

    if verbose:
        print (" ... rgb_image.shape = ",rgb_image.shape)
    rgb_image = rgb_image.swapaxes(0,2)
    rgb_image = rgb_image.swapaxes(1,2)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],num_periods_history,3,len(increments)))
    rgb_image = rgb_image.swapaxes(2,3)
    forecast = forecast.reshape(forecast.shape[0],1)

    if verbose:
        print(" ... Labeled data successfully created ...")
    return rgb_image, forecast, dates, company_name


def generateExamples3layerGen(datearray,adjClose,first_history_index,num_periods_history,increments,output_incr='monthly',verbose=False):

    ###
    ### attempt to re-write using generators instead of for-loops
    ###
    # output_incr can be 'monthly', 'weekly', or 'daily'

    # make certain 'increments' is sorted in ascending order
    increments.sort()

    dates = []
    rgb_image = np.array([])

    first_index = max( int(first_history_index), int(increments[-1]*num_periods_history) )

    def gen_rgb_row(jdate,datearray,adjClose,num_periods_history,increments,output_incr='monthly',verbose=False):

        #print("jdate, datearray[jdate], increments = ", jdate, datearray[jdate], increments)

        output_test = True

        if output_test == True:

            # create a row for each company

            _quotes_image = np.zeros((adjClose.shape[0], num_periods_history+1, len(increments)), 'float')
            _quotes_image_gains = np.zeros((adjClose.shape[0], num_periods_history, len(increments)), 'float')
            valid_indices_count = np.zeros((adjClose.shape[0]), 'int')
            for i,iincr in enumerate(increments):

                _quotes_image[:,:,i] = adjClose[:,jdate-num_periods_history*iincr:jdate+1:iincr].copy()
                _quotes_image_gains[:,:,i] = _quotes_image[:,1:,i] / _quotes_image[:,:-1,i] - 1.
                if iincr==increments[0]:
                    invalid_quote_count = np.zeros((_quotes_image_gains.shape[0]), 'int')
                #if _quotes_image[:,0,i]==0.:

                _test = (_quotes_image_gains[:,0,i]==0.)
                invalid_quote_count[_test] += np.ones((adjClose.shape[0]), 'int')[_test]

                _test = (_quotes_image_gains[:,-1,i]==0.)
                invalid_quote_count[_test] += np.ones((adjClose.shape[0]), 'int')[_test]

                _test = (np.isnan(_quotes_image_gains[:,:,i]))
                _test2 = np.sum( _test == True, axis=1)
                invalid_quote_count[_test2] += np.ones((adjClose.shape[0]), 'int')[_test2]

                _test = (np.isinf(_quotes_image_gains[:,:,i]))
                _test2 = np.sum( _test == True, axis=1)
                invalid_quote_count[_test2] += np.ones((adjClose.shape[0]), 'int')[_test2]

                invalid_indices = np.arange(adjClose.shape[0])[invalid_quote_count != 0]
                valid_indices_count[invalid_indices] += 1

            # save as gain/loss for period (drop first value)
            valid_indices = np.arange(adjClose.shape[0])[valid_indices_count == 0]
            rgb_row = _quotes_image_gains[valid_indices,:,:]
            _company_names = np.arange(adjClose.shape[0])[valid_indices]
            _forecasts = adjClose[:,jdate+22] / adjClose[:,jdate] - 1.
            _forecasts = _forecasts[valid_indices]
            _dates = [datearray[jdate] for i in range(_forecasts.shape[0])]

            #print('- 0 - rgb_row.shape = ', rgb_row.shape)
            #rgb_row = np.hstack((rgb_row,rgb_row,rgb_row))
            #print('- 1 - rgb_row.shape = ', rgb_row.shape)
            #rgb_row = rgb_row.swapaxes(2,3)
            #print('- 2 - rgb_row.shape = ', rgb_row.shape)
            #rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],rgb_row.shape[1],rgb_row.shape[2],1))
            #print('- 0 - datearray[jdate], _forecasts.shape = ', datearray[jdate], _forecasts.shape)
            yield rgb_row, _forecasts, _dates, _company_names

    dates_list = []
    for jdate in range(first_index, adjClose.shape[1]-22):
        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day
        if output_incr == 'monthly':
            output_test = jdate == first_history_index or \
                          datearray[jdate].year > datearray[jdate-1].year or \
                          datearray[jdate].month > datearray[jdate-1].month
        if output_incr == 'midmonth':
            output_test = jdate == first_history_index \
            or datearray[jdate].weekday() < datearray[jdate-1].weekday() \
            and (12 <= datearray[jdate].day < 19)
        elif output_incr == 'weekly':
            output_test = jdate == first_history_index or \
                          datearray[jdate].weekday() < datearray[jdate-1].weekday()
        elif output_incr == 'daily':
            output_test = True

        if output_test is True:
            dates_list.append(jdate)
        '''
            if verbose:
                print(" ... date to process : ", datearray[jdate])
        '''

    if verbose:
        print("first_index = ", first_index)
        print("dates_list first,last,len = ", dates_list[0],dates_list[-1],len(dates_list))
        print("datearray.shape = ", len(datearray))
        print("adjClose.shape = ", adjClose.shape)

    for jdate in dates_list:

        # create a row for each company
        num_companies = 0
        summ = np.zeros((num_periods_history,len(increments)),'float')
        sumSquared = np.zeros((num_periods_history,len(increments)),'float')

        for rgb_row, _forecast, _dates, _company_name in gen_rgb_row(jdate,
                                                                     datearray,
                                                                     adjClose,
                                                                     num_periods_history,
                                                                     increments,
                                                                     output_incr='monthly',
                                                                     verbose=False):

            num_companies = rgb_row.shape[0]

            rgb_row = np.dstack((rgb_row, rgb_row, rgb_row))
            for i,iincr in enumerate(increments):
                summ = rgb_row[:,:len(increments),i].sum(axis=-1)
                sumSquared = rgb_row[:,:len(increments),i].sum(axis=-1)**2
                #print (" ... 1. rgb_row.shape = ",rgb_row.shape, rgb_row[int(rgb_row.shape[0]/2),:,:])
                #print (" ... 1. _forecast.shape = ",np.array(_forecast).shape)
                #print (" ... 1. _dates.shape = ",np.array(_dates).shape)
                #print (" ... 1. _company_name.shape = ",_company_name.shape)

                #print (" ... 2. summ.shape = ",summ.shape)
                #print (" ... 3. sumSquared.shape = ",sumSquared.shape)
                avg_forallcompanies = summ / num_companies
                var_forallcompanies = (sumSquared - (summ**2)/num_companies)/(num_companies-1)

                #print (" ... 4. avg_forallcompanies.shape = ",avg_forallcompanies.reshape(avg_forallcompanies.shape[0],1).shape)
                #print (" ... 4. var_forallcompanies.shape = ",var_forallcompanies.shape)
                #print (" ... 4. rgb_row[:,:,i+len(increments)].shape = ",rgb_row[:,:,i+len(increments)].shape)

                rgb_row[:,:,i+len(increments)] -= avg_forallcompanies.reshape(avg_forallcompanies.shape[0],1)
                rgb_row[:,:,i+len(increments)*2] = var_forallcompanies.reshape(np.array(_forecast).shape[0],1)

            try:
                rgb_image = np.vstack((rgb_image, rgb_row))
                forecast = np.hstack((forecast, _forecast))
                dates = np.hstack((dates, _dates))
                company_name = np.hstack((company_name, _company_name))
                #print (" ... 5. rgb_image.shape = ",rgb_image.shape)
                #print (" ... 5. forecast.shape = ",forecast.shape)
            except:
                # need to initialize rgb_image
                # indices order will be periods, iCompany
                rgb_image = rgb_row
                forecast = _forecast
                dates = _dates
                company_name = _company_name


        '''
        if verbose:
            if datearray[jdate].year > datearray[jdate-1].year:
                #print("  ... jdate, iCompany, rgb_row.shape = ", jdate,iCompany, rgb_row.shape)
                print("  ... progress ... finished ", datearray[jdate-1].year)
        '''

    if verbose:
        print (" ... rgb_image.shape = ",rgb_image.shape)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],num_periods_history,3,len(increments)))
    rgb_image = rgb_image.swapaxes(2,3)
    forecast = forecast.reshape(forecast.shape[0],1)

    if verbose:
        print(" ... Labeled data successfully created ...")
    return rgb_image, forecast, dates, company_name



def generateExamples3layerForDate(predict_date,
                                  datearray,
                                  adjClose,
                                  num_periods_history,
                                  increments,
                                  output_incr='monthly',
                                  verbose=False):

    # output_incr can be 'monthly', 'weekly', or 'daily'

    # make certain 'increments' is sorted in ascending order
    increments.sort()

    invalid_count = 0
    dates = []
    rgb_image = np.array([])

    first_index = np.argmin(np.abs(np.array(datearray)-predict_date))

    for jdate in range(first_index, first_index +1):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day
        output_test = True

        if datearray[jdate].year > datearray[jdate-1].year:
            if verbose:
                print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

        if output_test == True:

            # create a row for each company
            num_companies = 0
            summ = np.zeros((num_periods_history,len(increments)),'float')
            sumSquared = np.zeros((num_periods_history,len(increments)),'float')

            for iCompany in range(adjClose.shape[0]):

                _quotes_image = np.zeros((num_periods_history+1,len(increments)),'float')
                for i, iincr in enumerate(increments):

                    _quotes_image[:,i] = adjClose[iCompany,jdate-num_periods_history*iincr:jdate+1:iincr].copy()

                    if iincr==increments[0]:
                        invalid_quote_count = 0
                    if _quotes_image[0,i]==0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    if _quotes_image[-1,i] == 0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    nantest = _quotes_image[:,i]
                    if nantest[np.isnan(nantest)].shape[0] > 0 or nantest[np.isinf(nantest)].shape[0] > 0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    if verbose:
                        print("invalid quotes for iCompany,date : ",iCompany, datearray[jdate])
                    continue

                # save as gain/loss for period (drop first value)
                _quotes_image = _quotes_image[1:,:] / _quotes_image[:-1,:] - 1.

                for i,iincr in enumerate(increments):

                    if iincr==increments[0]:
                        invalid_quote_count = 0

                    _quote_image_list = list(_quotes_image[:,i])
                    _quote_image_list_invalid = _substringfinder(_quote_image_list, [0.0,0.0])

                    if _quote_image_list_invalid>0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    continue

                #print("_quotes_image[:,0] = ",_quotes_image[:,0])
                #print("_quotes_image[:,1] = ",_quotes_image[:,1])
                #print("_quotes_image[:,2] = ",_quotes_image[:,2])

                # combine and reshape
                rgb_row = np.dstack((_quotes_image,_quotes_image,_quotes_image)).swapaxes(1,2)
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],3*len(increments),1))
                #print('rgb_row.shape = ', rgb_row.shape)
                #print('rgb_row = ', rgb_row)

                try:
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    forecast = np.hstack((forecast,adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    forecast = adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                summ += rgb_row[:,:len(increments),0]
                sumSquared += rgb_row[:,:len(increments),0]**2
                #print (" ... rgb_row.shape = ",rgb_row.shape,rgb_row)

            #print (" ... summ = ",summ)
            #print (" ... sumSquared = ",sumSquared)
            avg_forallcompanies = summ / num_companies
            var_forallcompanies = (sumSquared - (summ**2)/num_companies)/(num_companies-1)
            #print (" ... avg_forallcompanies = ",avg_forallcompanies)
            #print (" ... var_forallcompanies = ",var_forallcompanies)
            #print (" ... rgb_image.shape = ",rgb_image.shape)
            #print (" ... rgb_row.shape = ",rgb_row.shape)

            for iCompany in range(1,num_companies+1):
                rgb_image[:,len(increments):2*len(increments),-iCompany] -= avg_forallcompanies
                rgb_image[:,2*len(increments):,-iCompany] = var_forallcompanies
            #print('gain/loss rgb_image[:,:3,-1] = ',rgb_image[:,:3,-1])
            #print('wo mean   rgb_image[:,3:6,-1] = ',rgb_image[:,3:6,-1])
            #print('variance  rgb_image[:,6:9,-1] = ',rgb_image[:,6:9,-1])

    if verbose:
        print (" ... rgb_image.shape = ",rgb_image.shape)
    rgb_image = rgb_image.swapaxes(0,2)
    rgb_image = rgb_image.swapaxes(1,2)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],num_periods_history,3,len(increments)))
    rgb_image = rgb_image.swapaxes(2,3)
    forecast = forecast.reshape(forecast.shape[0],1)

    if verbose:
        print(" ... Labeled data successfully created ...")
    return rgb_image, forecast, dates, company_name


def generatePredictionInput3layer(predict_date,datearray,adjClose,first_history_index,num_periods_history,increments,output_incr='monthly',verbose=False):

    # predict_date is a single date for which input will be re-formatted to make predictions using DL

    # make certain 'increments' is sorted in ascending order
    increments.sort()

    invalid_count = 0
    dates = []
    rgb_image = np.array([])


    for jdate in range(adjClose.shape[1]):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day

        output_test = False

        if datearray[jdate] == predict_date:
            output_test = True

        if datearray[jdate].year > datearray[jdate-1].year and verbose==True:
            print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

        if output_test == True:

            # create a row for each company
            num_companies = 0
            summ = np.zeros((num_periods_history,len(increments)),'float')
            sumSquared = np.zeros((num_periods_history,len(increments)),'float')

            for iCompany in range(adjClose.shape[0]):

                _quotes_image = np.zeros((num_periods_history+1,len(increments)),'float')
                for i,iincr in enumerate(increments):

                    _quotes_image[:,i] = adjClose[iCompany,jdate-num_periods_history*iincr:jdate+iincr:iincr].copy()

                    if iincr==increments[0]:
                        invalid_quote_count = 0
                    if _quotes_image[0,i]==0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    if _quotes_image[-1,i] == 0.:
                        invalid_count += 1
                        invalid_quote_count += 1

                    nantest = _quotes_image[:,i]
                    if nantest[np.isnan(nantest)].shape[0] > 0 or nantest[np.isinf(nantest)].shape[0] > 0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    if verbose:
                        print("invalid quotes for iCompany,date : ",iCompany, datearray[jdate])
                    continue

                # save as gain/loss for period (drop first value)
                _quotes_image = _quotes_image[1:,:] / _quotes_image[:-1,:] - 1.

                for i,iincr in enumerate(increments):

                    if iincr==increments[0]:
                        invalid_quote_count = 0

                    _quote_image_list = list(_quotes_image[:,i])
                    _quote_image_list_invalid = _substringfinder(_quote_image_list, [0.0,0.0])

                    if _quote_image_list_invalid>0:
                        invalid_count += 1
                        invalid_quote_count += 1

                if invalid_quote_count > 0:
                    continue

                # combine and reshape
                rgb_row = np.dstack((_quotes_image,_quotes_image,_quotes_image)).swapaxes(1,2)
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],3*len(increments),1))

                try:
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                summ += rgb_row[:,:len(increments),0]
                sumSquared += rgb_row[:,:len(increments),0]**2

            avg_forallcompanies = summ / num_companies
            var_forallcompanies = (sumSquared - (summ**2)/num_companies)/(num_companies-1)

            for iCompany in range(1,num_companies+1):
                rgb_image[:,len(increments):2*len(increments),-iCompany] -= avg_forallcompanies
                rgb_image[:,2*len(increments):,-iCompany] += var_forallcompanies

    rgb_image = rgb_image.swapaxes(0,2)
    rgb_image = rgb_image.swapaxes(1,2)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],num_periods_history,3,len(increments)))
    rgb_image = rgb_image.swapaxes(2,3)

    if verbose:
        print(" ... Labeled data successfully created ...")
    return rgb_image, dates, company_name


"""
def generatePredictionInput(datearray,adjClose,first_history_index,num_periods_history,increments,output_incr='monthly'):

    # output_incr can be 'monthly', 'weekly', or 'daily'

    shortest_incr = increments[0]
    middle_incr = increments[1]
    long_incr = increments[2]

    invalid_count = 0
    dates = []
    rgb_image = np.array([])

    first_index = max( int(first_history_index), int(long_incr*num_periods_history) )

    for jdate in range(first_index,adjClose.shape[1]):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day
        if output_incr == 'monthly':
            output_test = jdate == first_history_index or datearray[jdate].year > datearray[jdate-1].year or datearray[jdate].month > datearray[jdate-1].month
        if output_incr == 'midmonth':
            output_test = jdate == first_history_index \
            or datearray[jdate].weekday() < datearray[jdate-1].weekday() \
            and (12 <= datearray[jdate].day < 19)
        elif output_incr == 'weekly':
            output_test = jdate == first_history_index or datearray[jdate].weekday() < datearray[jdate-1].weekday()
        elif output_incr == 'daily':
            output_test = True

        if datearray[jdate].year > datearray[jdate-1].year:
            print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

        if output_test == True:

            # create a row for each company
            num_companies = 0
            avg_forallcompanies = np.zeros((num_periods_history,3),'float')
            for iCompany in range(adjClose.shape[0]):

                r_array = adjClose[iCompany,jdate-num_periods_history*shortest_incr:jdate+shortest_incr:shortest_incr].copy()
                g_array = adjClose[iCompany,jdate-num_periods_history*middle_incr:jdate+middle_incr:middle_incr].copy()
                b_array = adjClose[iCompany,jdate-num_periods_history*long_incr:jdate+long_incr:long_incr].copy()

                if r_array[0]==0. or g_array[0]==0. or b_array[0]==0.:
                    invalid_count += 1
                    continue

                # filter out series without valid quotes at beginning
                r_list = list(r_array/r_array[0])
                g_list = list(g_array/g_array[0])
                b_list = list(b_array/b_array[0])

                r_count_invalid = _substringfinder(r_list, [1.0,1.0])
                g_count_invalid = _substringfinder(g_list, [1.0,1.0])
                b_count_invalid = _substringfinder(b_list, [1.0,1.0])

                if r_count_invalid>0 or g_count_invalid>0 or b_count_invalid>0:
                    invalid_count += 1
                    continue

                if r_array[:-1].min()==0. or g_array[:-1].min()==0. or b_array[:-1].min()==0.:
                    invalid_count += 1
                    continue

                # save as gain/loss for period (drop first value)
                r_array = r_array[1:] / r_array[:-1] - 1.
                g_array = g_array[1:] / g_array[:-1] - 1.
                b_array = b_array[1:] / b_array[:-1] - 1.

                # combine and reshpae
                rgb_row = np.vstack((r_array,g_array,b_array,r_array,g_array,b_array))
                rgb_row = np.swapaxes(rgb_row,0,1)
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],6,1))

                try:
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    forecast = np.hstack((forecast,adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    forecast = adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                avg_forallcompanies += rgb_row[:,3:,0]

            avg_forallcompanies /= num_companies
            for iCompany in range(1,num_companies+1):
                rgb_image[:,3:,-iCompany] -= avg_forallcompanies

    rgb_image = rgb_image.swapaxes(0,2)
    rgb_image = rgb_image.swapaxes(1,2)
    rgb_image = rgb_image.reshape((rgb_image.shape[0],20,3,2))
    forecast = forecast.reshape(forecast.shape[0],1)

    print(" ... Labeled data successfully created ...")
    return rgb_image, dates, company_name
"""

"""
def generateExamples_randomValidate(datearray,adjClose,first_history_index,num_periods_history,increments,output_incr='monthly',validate_pct=0.1):

    # output_incr can be 'monthly', 'weekly', or 'daily'

    shortest_incr = increments[0]
    middle_incr = increments[1]
    long_incr = increments[2]

    invalid_count = 0
    dates = []
    rgb_image = []

    for jdate in range(int(first_history_index),adjClose.shape[1]-22):

        # create 1 row per company in adjClose
        # create 1 image at beginning of each month, week, or day
        if output_incr == 'monthly':
            output_test = jdate == first_history_index or datearray[jdate].year > datearray[jdate-1].year or datearray[jdate].month > datearray[jdate-1].month
        elif output_incr == 'weekly':
            output_test = jdate == first_history_index or datearray[jdate].weekday() < datearray[jdate-1].weekday()
        elif output_incr == 'daily':
            output_test = True

        if output_test == True:

            if datearray[jdate].year > datearray[jdate-1].year:
                print(" ... processing date : ", datearray[jdate]," ... invalid/valid counts = ",invalid_count,", ",rgb_image.shape[-1])

            # create a row for each company
            num_companies = 0
            avg_forallcompanies = np.zeros((num_periods_history,3),'float')
            for iCompany in range(adjClose.shape[0]):

                r_array = adjClose[iCompany,jdate-num_periods_history*shortest_incr:jdate+shortest_incr:shortest_incr].copy()
                g_array = adjClose[iCompany,jdate-num_periods_history*middle_incr:jdate+middle_incr:middle_incr].copy()
                b_array = adjClose[iCompany,jdate-num_periods_history*long_incr:jdate+long_incr:long_incr].copy()

                # filter out series without valid quotes at beginning
                r_list = list(r_array/r_array[0])
                g_list = list(g_array/g_array[0])
                b_list = list(b_array/b_array[0])

                r_count_invalid = _substringfinder(r_list, [1.0,1.0])
                g_count_invalid = _substringfinder(g_list, [1.0,1.0])
                b_count_invalid = _substringfinder(b_list, [1.0,1.0])

                if r_count_invalid>0 or g_count_invalid>0 or b_count_invalid>0:
                    invalid_count += 1
                    continue

                # save as gain/loss for period (drop first value)
                r_array = r_array[1:] / r_array[:-1] - 1.
                g_array = g_array[1:] / g_array[:-1] - 1.
                b_array = b_array[1:] / b_array[:-1] - 1.

                # combine and reshpae
                rgb_row = np.vstack((r_array,g_array,b_array,r_array,g_array,b_array))
                rgb_row = np.swapaxes(rgb_row,0,1)
                rgb_row = np.reshape(rgb_row,(rgb_row.shape[0],6,1))

                try:
                    # print(" ... rgb_image and rgb_row shapes = ", np.array(rgb_image).shape, np.array(rgb_row).shape)
                    rgb_image = np.dstack((rgb_image,rgb_row))
                    forecast = np.hstack((forecast,adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate]-1.))
                    dates.append( datearray[jdate] )
                    company_name.append( iCompany )
                except:
                    # need to initialize rgb_image
                    # indices order will be periods, iCompany
                    rgb_image = rgb_row
                    forecast = adjClose[iCompany,jdate+22]/adjClose[iCompany,jdate] - 1.
                    dates.append( datearray[jdate] )
                    company_name = [ iCompany ]

                num_companies += 1

                avg_forallcompanies += rgb_row[:,3:,0]

            avg_forallcompanies /= num_companies
            for iCompany in range(1,num_companies+1):
                rgb_image[:,3:,-iCompany] -= avg_forallcompanies

            TorV = np.random.choice(np.array(['train','validate']),
                                    p=[1.-validate_pct,validate_pct])
            for iCompany in range(1,num_companies+1):
                if TorV == 'train':
                    try:
                        Xtrain = np.dstack((Xtrain,rgb_row))
                        Ytrain = np.hstack((Ytrain,forecast))
                        dates_train.append( dates )
                        company_train.append( company_name )
                    except:
                        Xtrain = rgb_row
                        Ytrain = forecast
                        dates_train = dates
                        company_train = company_name
                elif TorV == 'validate':
                    try:
                        Xvalidate = np.dstack((Xvalidate,rgb_row))
                        Yvalidate = np.hstack((Yvalidate,forecast))
                        dates_validate.append( dates )
                        company_validate.append( company_name )
                    except:
                        Xvalidate = rgb_row
                        Yvalidate = forecast
                        dates_validate = dates
                        company_validate = company_name

    Xtrain = Xtrain.swapaxes(0,2)
    Xtrain = Xtrain.swapaxes(1,2)
    Xtrain = Xtrain.reshape((Xtrain.shape[0],20,3,2))
    Ytrain = Ytrain.reshape(Ytrain.shape[0],1)

    Xvalidate = Xvalidate.swapaxes(0,2)
    Xvalidate = Xvalidate.swapaxes(1,2)
    Xvalidate = Xvalidate.reshape((Xvalidate.shape[0],20,3,2))
    Yvalidate = Yvalidate.reshape(Yvalidate.shape[0],1)

    print("training & validation examples created successfully")

    return Xtrain, Ytrain, Xvalidate, Yvalidate
"""


def generateExamples_randomValidate(datearray,adjClose,first_history_index,
                                    num_periods_history,increments,
                                    output_incr='monthly',validate_pct=0.1):


    # output_incr can be 'monthly', 'weekly', or 'daily'

    # reformat into examples without regard to training or validation
    X, Y, dates, company_names = generateExamples(datearray,
                                                  adjClose,
                                                  first_history_index,
                                                  num_periods_history,
                                                  increments,
                                                  output_incr=output_incr)

    x_input_shape = X.shape

    X = X.reshape( X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
    X = X.swapaxes(0,1)
    X = X.swapaxes(1,2)

    Y = Y[:,0]

    # split into training and validation subsets. Random months move to validation
    nTrain = 0
    nValidate = 0
    for i,idate in enumerate(dates):
        if i == 0 or idate != dates[i-1]:
            TorV = np.random.choice(np.array(['train','validate']),
                                    p=[1.-validate_pct,validate_pct])
        if TorV == 'train':
            if nTrain==0:
                Xtrain = X[...,i]
                Ytrain = Y[...,i]
                dates_train = dates[i]
                company_train = company_names[i]
                nTrain += 1
            else:
                Xtrain = np.dstack((Xtrain,X[...,i]))
                Ytrain = np.hstack((Ytrain,Y[...,i]))
                dates_train = np.hstack((dates_train,dates[i]))
                company_train = np.hstack((company_train,company_names[i]))
                #company_train.append( company_names[i] )
                nTrain += 1
        elif TorV == 'validate':
            if nValidate==0:
                Xvalidate = X[...,i]
                Yvalidate = Y[...,i]
                dates_validate = dates[i]
                company_validate = company_names[i]
                nValidate += 1
            else:
                Xvalidate = np.dstack((Xvalidate,X[...,i]))
                Yvalidate = np.hstack((Yvalidate,Y[...,i]))
                dates_validate = np.hstack((dates_validate,dates[i]))
                company_validate = np.hstack((company_validate,company_names[i]))
                #company_validate.append( company_names[i] )
                nValidate += 1

    Xtrain = Xtrain.swapaxes(0,2)
    Xtrain = Xtrain.swapaxes(1,2)
    Xtrain = Xtrain.reshape( Xtrain.shape[0], x_input_shape[1], x_input_shape[2], x_input_shape[3])

    Ytrain = Ytrain.reshape( Ytrain.shape[0], 1 )

    Xvalidate = Xvalidate.swapaxes(0,2)
    Xvalidate = Xvalidate.swapaxes(1,2)
    Xvalidate = Xvalidate.reshape( Xvalidate.shape[0], x_input_shape[1], x_input_shape[2], x_input_shape[3])

    Yvalidate = Yvalidate.reshape( Yvalidate.shape[0], 1 )

    print(" ... Labeled data successfully split into training and validation subsets ...")
    return ( Xtrain, Ytrain, dates_train, company_train,
            Xvalidate, Yvalidate, dates_validate, company_validate )


def normcorrcoef(a,b):
    return np.correlate(a,b)/np.sqrt(np.correlate(a,a)*np.correlate(b,b))[0]


def interpolate(self, method='linear'):
    """
    Interpolate missing values (after the first valid value)
    Parameters
    ----------
    method : {'linear'}
    Interpolation method.
    Time interpolation works on daily and higher resolution
    data to interpolate given length of interval
    Returns
    -------
    interpolated : Series
    from-- https://github.com/wesm/pandas/blob/master/pandas/core/series.py
    edited to keep only 'linear' method
    Usage: infill NaN values with linear interpolated values
    """
    inds = np.arange(len(self))
    values = np.array(self.copy())
    invalid_bool = np.isnan(values)
    valid = np.ones((len(self)),'int')
    valid[ invalid_bool==True ] = 0
    invalid = 1 - valid
    firstIndex = valid.argmax()
    lastIndex = valid.shape[0]-valid[::-1].argmax()
    #print("(firstIndex, lastIndex) = ",(firstIndex,lastIndex))
    #valid = valid[firstIndex:lastIndex]
    #invalid = invalid[firstIndex:lastIndex]
    valid = valid[valid >= firstIndex]
    valid = valid[valid <= lastIndex]
    invalid = invalid[invalid >= firstIndex]
    invalid = invalid[invalid <= lastIndex]

    #inds = inds[firstIndex:]
    result = values.copy()
    #result[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1] = np.interp(inds[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1], inds[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1],values[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1])
    if len(invalid[invalid==1]) > 0:
        result[invalid==1] = np.interp(inds[invalid==1], inds[valid==1],values[valid==1])

    return result

#----------------------------------------------
def cleantobeginning(self):
    """
    Copy missing values (to all dates prior the first valid value)

    Usage: infill NaN values at beginning with copy of first valid value
    """
    inds = np.arange(len(self))
    values = self.copy()
    invalid_bool = np.isnan(values)
    valid = np.ones((len(self)),'int')
    valid[ invalid_bool==True ] = 0
    invalid = 1 - valid
    firstIndex = valid.argmax()
    for i in range(firstIndex):
        values[i]=values[firstIndex]
    return values

#----------------------------------------------

def cleantoend(self):
    """
    Copy missing values (to all dates after the last valid value)

    Usage: infill NaN values at end with copy of last valid value
    """
    # reverse input 1D array and use cleantobeginning
    reverse = self[::-1]
    reverse = cleantobeginning(reverse)
    return reverse[::-1]

#----------------------------------------------

def clean_signal(array1D,symbol_name):
    ### clean input signals (again)
    quotes_before_cleaning = array1D.copy()
    adjClose = interpolate( array1D )
    adjClose = cleantobeginning( adjClose )
    adjClose = cleantoend( adjClose )
    adjClose_changed = False in (adjClose==quotes_before_cleaning)
    print("   ... inside PortfolioPerformanceCalcs ... symbol, did cleaning change adjClose? ", symbol_name, adjClose_changed)
    return adjClose

#----------------------------------------------

def cleanspikes(x,periods=20,stddevThreshold=5.0):
    # remove outliers from gradient of x (in 2 directions)
    x_clean = np.array(x).copy()
    test = np.zeros(x.shape[0],'float')
    #gainloss_f = np.ones((x.shape[0]),dtype=float)
    #gainloss_r = np.ones((x.shape[0]),dtype=float)
    #print gainloss_f[1:],x[1:].shape,x[:-1].shape
    #print " ...inside cleanspikes... ", x[1:].shape, x[:-1].shape
    #gainloss_f[1:] = x[1:] / x[:-1]
    #gainloss_r[:-1] = x[:-1] / x[1:]
    gainloss_f = x[1:] / x[:-1]
    gainloss_r = x[:-1] / x[1:]
    valid_f = gainloss_f[gainloss_f != 1.]
    valid_f = valid_f[~np.isnan(valid_f)]
    Stddev_f = np.std(valid_f) + 1.e-5
    valid_r = gainloss_r[gainloss_r != 1.]
    valid_r = valid_r[~np.isnan(valid_r)]
    Stddev_r = np.std(valid_r) + 1.e-5

    forward_test = gainloss_f/Stddev_f - np.median(gainloss_f/Stddev_f)
    reverse_test = gainloss_r/Stddev_r - np.median(gainloss_r/Stddev_r)

    test[:-1] += reverse_test
    test[1:] += forward_test

    x_clean[ test > stddevThreshold ] = np.nan

    """
    for i in range( 1,x.shape[0]-2 ):
         minx = max(0,i-periods/2)
         maxx = min(x.shape[0],i+periods/2)
         #Stddev_f = np.std(gainloss_f[minx:maxx]) + 1.e-5
         #Stddev_r = np.std(gainloss_r[minx:maxx]) + 1.e-5
         if gainloss_f[i-1]/Stddev_f > stddevThreshold and gainloss_r[i]/Stddev_r > stddevThreshold:
            x_clean[i] = np.nan
    """
    return x_clean

#----------------------------------------------

def percentileChannel(x,minperiod,maxperiod,incperiod,lowPct,hiPct):
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros(len(x),dtype=float)
    maxchannel = np.zeros(len(x),dtype=float)
    for i in range(len(x)):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            if len(x[minx:i]) < 1:
                minchannel[i] = minchannel[i] + x[i]
                maxchannel[i] = maxchannel[i] + x[i]
                divisor += 1
            else:
                minchannel[i] = minchannel[i] + np.percentile(x[minx:i+1],lowPct)
                maxchannel[i] = maxchannel[i] + np.percentile(x[minx:i+1],hiPct)
                divisor += 1
        minchannel[i] /= divisor
        maxchannel[i] /= divisor
    return minchannel,maxchannel
#----------------------------------------------
def percentileChannel_2D(x,minperiod,maxperiod,incperiod,lowPct,hiPct):
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    maxchannel = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            if len(x[0,minx:i]) < 1:
                minchannel[:,i] = minchannel[:,i] + x[:,i]
                maxchannel[:,i] = maxchannel[:,i] + x[:,i]
                divisor += 1
            else:
                minchannel[:,i] = minchannel[:,i] + np.percentile(x[:,minx:i+1],lowPct,axis=-1)
                maxchannel[:,i] = maxchannel[:,i] + np.percentile(x[:,minx:i+1],hiPct,axis=-1)
                divisor += 1
        minchannel[:,i] /= divisor
        maxchannel[:,i] /= divisor
    print(" minperiod,maxperiod,incperiod = ", minperiod,maxperiod,incperiod)
    print(" lowPct,hiPct = ", lowPct,hiPct)
    print(" x min,mean,max = ", x.min(),x.mean(),x.max())
    print(" divisor = ", divisor)
    return minchannel,maxchannel


#----------------------------------------------
def dpgchannel(x,minperiod,maxperiod,incperiod):
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros(len(x),dtype=float)
    maxchannel = np.zeros(len(x),dtype=float)
    for i in range(len(x)):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            if len(x[minx:i]) < 1:
                minchannel[i] = minchannel[i] + x[i]
                maxchannel[i] = maxchannel[i] + x[i]
                divisor += 1
            else:
                minchannel[i] = minchannel[i] + min(x[minx:i+1])
                maxchannel[i] = maxchannel[i] + max(x[minx:i+1])
                divisor += 1
        minchannel[i] /= divisor
        maxchannel[i] /= divisor
    return minchannel,maxchannel
#----------------------------------------------
def dpgchannel_2D(x,minperiod,maxperiod,incperiod):
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    maxchannel = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            if len(x[0,minx:i]) < 1:
                minchannel[:,i] = minchannel[:,i] + x[:,i]
                maxchannel[:,i] = maxchannel[:,i] + x[:,i]
                divisor += 1
            else:
                minchannel[:,i] = minchannel[:,i] + np.min(x[:,minx:i+1],axis=-1)
                maxchannel[:,i] = maxchannel[:,i] + np.max(x[:,minx:i+1],axis=-1)
                divisor += 1
        minchannel[:,i] /= divisor
        maxchannel[:,i] /= divisor
    return minchannel,maxchannel
#----------------------------------------------
def selfsimilarity(hi,lo):

    from scipy.stats import percentileofscore
    HminusL = hi-lo

    periods = 10
    SMS = np.zeros( (hi.shape[0]), dtype=float)
    for i in range( hi.shape[0] ):
        minx = max(0,i-periods)
        SMS[i] = np.sum(HminusL[minx:i+1],axis=-1)

    # find the 10-day range (incl highest high and lowest low)
    range10day = MoveMax(hi,10) - MoveMin(lo,10)

    # normalize
    SMS /= range10day

    # compute quarterly (60-day) SMA
    SMS = SMA(SMS,60)

    # find percentile rank
    movepctrank = np.zeros( (hi.shape[0]), dtype=float)
    for i in range( hi.shape[0] ):
        minx = max(0,i-periods)
        movepctrank[i] = percentileofscore(SMS[minx:i+1],SMS[i])

    return movepctrank

#----------------------------------------------
def jumpTheChannelTest(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):
    ###
    ### compute linear trend in upper and lower channels and compare
    ### actual stock price to forecast range
    ### return pctChannel for each stock
    ### calling function will use pctChannel as signal.
    ### - e.g. negative pctChannel is signal that down-trend begins
    ### - e.g. more than 100% pctChanel is sgnal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    import warnings
    warnings.simplefilter('ignore', np.RankWarning)

    pctChannel = np.zeros( (x.shape[0]), 'float' )
    # calculate linear trend over 'numdaysinfit' with 'offset'
    minchannel,maxchannel = dpgchannel(x,minperiod,maxperiod,incperiod)
    minchannel_trenddata = minchannel[-(numdaysinfit+offset):-offset]
    regression = np.polyfit(list(range(-(numdaysinfit+offset),-offset)), minchannel_trenddata, 1)
    minchannel_trend = regression[-1]
    maxchannel_trenddata = maxchannel[-(numdaysinfit+offset):-offset]
    regression = np.polyfit(list(range(-(numdaysinfit+offset),-offset)), maxchannel_trenddata, 1)
    maxchannel_trend = regression[-1]
    pctChannel = (x[-1]-minchannel_trend) / (maxchannel_trend-minchannel_trend)

    # calculate the stdev over the period
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.
    gainloss_std = np.std( gainloss_period )

    # calculate the current quote as number of stdevs above or below trend
    currentMidChannel = (maxchannel_trenddata+minchannel_trend)/2.
    numStdDevs = (x[-1]/currentMidChannel[-1]-1.) / gainloss_std

    '''
    print "pctChannel = ", pctChannel
    print "gainloss_period = ", gainloss_period
    print "gainloss_cumu = ", gainloss_cumu
    print "gainloss_std = ", gainloss_std
    print "currentMidChannel = ", currentMidChannel[-1]
    print "numStdDevs = ", numStdDevs
    '''

    return pctChannel, gainloss_cumu, gainloss_std, numStdDevs

#----------------------------------------------
def recentChannelFit(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):
    ###
    ### compute cumulative gain over fitting period and number of
    ### ratio of current quote to fitted trend. Rescale based on std dev
    ### of residuals during fitting period.
    ### - e.g. negative pctChannel is signal that down-trend begins
    ### - e.g. more than 100% pctChanel is sgnal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    import warnings
    warnings.simplefilter('ignore', np.RankWarning)

    ##pctChannel = np.zeros( (x.shape[0]), 'float' )
    # calculate linear trend over 'numdaysinfit' with 'offset'
    minchannel,maxchannel = dpgchannel(x,minperiod,maxperiod,incperiod)
    if offset == 0:
        minchannel_trenddata = minchannel[-(numdaysinfit+offset):]
        '''
        print "numdaysinfit = ", numdaysinfit
        print "offset = ", offset
        print "len(x) = ", len(x)
        print "len(minchannel) = ", len(minchannel)
        print "most recent quote = ", x[-1]
        print "quote[-offset:] = ", x[-offset:]
        print "quote[-(numdaysinfit+offset)+1:-offset+1] = ", x[-(numdaysinfit+offset):]
        print "len(quote[-(numdaysinfit+offset)+1:-offset+1]) = ", len(x[-(numdaysinfit+offset):])
        print "length of days = ", len(range(-(numdaysinfit+offset)+1,-offset+1))
        print "relative days = ", range(-(numdaysinfit+offset)+1,-offset+1)
        print "length of quotes = ", len(minchannel_trenddata)
        print "quotes = ", minchannel_trenddata
        '''
        regression1 = np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), minchannel_trenddata, 1)
        minchannel_trend = regression1[-1]
        maxchannel_trenddata = maxchannel[-(numdaysinfit+offset):]
        regression2 = np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), maxchannel_trenddata, 1)
    else:
        minchannel_trenddata = minchannel[-(numdaysinfit+offset)+1:-offset+1]
        '''
        print "numdaysinfit = ", numdaysinfit
        print "offset = ", offset
        print "len(x) = ", len(x)
        print "len(minchannel) = ", len(minchannel)
        print "most recent quote = ", x[-1]
        print "quote[-offset:] = ", x[-offset:]
        print "quote[-(numdaysinfit+offset)+1:-offset+1] = ", x[-(numdaysinfit+offset)+1:-offset+1]
        print "len(quote[-(numdaysinfit+offset)+1:-offset+1]) = ", len(x[-(numdaysinfit+offset)+1:-offset+1])
        print "length of days = ", len(range(-(numdaysinfit+offset)+1,-offset+1))
        print "relative days = ", range(-(numdaysinfit+offset)+1,-offset+1)
        print "length of quotes = ", len(minchannel_trenddata)
        print "quotes = ", minchannel_trenddata
        '''
        regression1 = np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), minchannel_trenddata, 1)
        minchannel_trend = regression1[-1]
        maxchannel_trenddata = maxchannel[-(numdaysinfit+offset)+1:-offset+1]
        regression2 = np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), maxchannel_trenddata, 1)
    ##maxchannel_trend = regression2[-1]
    ##pctChannel = (x[-1]-minchannel_trend) / (maxchannel_trend-minchannel_trend)

    return regression1, regression2

#----------------------------------------------
def recentTrendAndStdDevs(x,datearray,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):

    ###
    ### compute linear trend in upper and lower channels and compare
    ### actual stock price to forecast range
    ### return pctChannel for each stock
    ### calling function will use pctChannel as signal.
    ### - e.g. numStdDevs < -1. is signal that down-trend begins
    ### - e.g. whereas  > 1.0 is signal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel for plotting
    lowerFit, upperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(upperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(lowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.
    #residuals = x[-numdaysinfit-offset:-offset+1] - midTrend
    #fitStdDev = np.std(residuals)
    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    #print ".....lowerFit, upperFit = ", lowerFit, upperFit
    #print ".....fitStdDev,currentUpper,currentLower,x[-1] = ", fitStdDev, currentUpper,currentLower,x[-1]
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    numStdDevs = currentResidual / fitStdDev

    # calculate gain or loss over the period
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    # different method for gainloss over period using slope
    gainloss_cumu = midTrend[-1] / midTrend[0] -1.

    pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)

    return gainloss_cumu, numStdDevs, pctChannel

#----------------------------------------------

def recentSharpeWithAndWithoutGap(x,numdaysinfit=28,numdaysinfit2=20, offset=3):

    from math import sqrt
    from scipy.stats import gmean

    ###
    ### - Cmpute sharpe ratio for recent prices with gap of 'offset' recent days
    ### - Compute 2nd sharpe ratio for recent prices recent days

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate sharpe with a gap
    # - 'numdaysinfit2' describes number of days over which to calculate sharpe without a gap
    # - 'offset'  describes number recent days to skip (e.g. the gap)

    # calculate gain or loss over the gapped period
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.

    # sharpe ratio in period with a gap
    sharpe_withGap = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )

    # calculate gain or loss over the period without a gap
    gainloss_period = x[-numdaysinfit2+1:] / x[-numdaysinfit2:-1]
    gainloss_period[np.isnan(gainloss_period)] = 1.

    # sharpe ratio in period wihtout a gap
    sharpe_withoutGap = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )

    # combine sharpe ratios compouted over 2 different periods
    # - use an angle of 33 degrees instead of 45 to give slightly more weight the the "no gap" sharpe
    crossplot_rotationAngle = 33. * np.pi/180.
    sharpe2periods = sharpe_withGap*np.sin(crossplot_rotationAngle) + sharpe_withoutGap*np.cos(crossplot_rotationAngle)

    print("sharpe with, without gap, combined = ", sharpe_withGap, sharpe_withoutGap, sharpe2periods)
    return sharpe2periods

#----------------------------------------------

def recentSharpeWithAndWithoutGap(x,numdaysinfit=504,offset_factor=.4):

    from math import sqrt
    from scipy.stats import gmean

    ###
    ### - Cmpute sharpe ratio for recent prices with gap of 'offset' recent days
    ### - Compute 2nd sharpe ratio for recent prices recent days

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate sharpe with a gap
    # - 'numdaysinfit2' describes number of days over which to calculate sharpe without a gap
    # - 'offset'  describes number recent days to skip (e.g. the gap)

    # calculate number of loops
    sharpeList = []
    for i in range(1,25):
        if i == 1:
            numdaysStart = numdaysinfit
            numdaysEnd = numdaysStart * offset_factor
        else:
            numdaysStart /= 2
            if numdaysStart/2 > 20:
                numdaysEnd = numdaysStart * offset_factor
            else:
                numdaysEnd = 0

        # calculate gain or loss over the gapped period
        numdays = numdaysStart - numdaysEnd
        offset = numdaysEnd
        if offset > 0:
            print("i,start,end = ", i, -(numdays+offset)+1, -offset+1)
            gainloss_period = x[-(numdays+offset)+1:-offset+1] / x[-(numdays+offset):-offset]
            gainloss_period[np.isnan(gainloss_period)] = 1.

            # sharpe ratio in period with a gap
            sharpe = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )
        else:
            print("i,start,end = ", i, -numdays+1, 0)
            # calculate gain or loss over the period without a gap
            gainloss_period = x[-numdays+1:] / x[-numdays:-1]
            gainloss_period[np.isnan(gainloss_period)] = 1.

            # sharpe ratio in period wihtout a gap
            sharpe = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )
        sharpeList.append(sharpe)
        if numdaysStart/2 < 20:
            break

    print("sharpeList = ", sharpeList)
    sharpeList = np.array(sharpeList)
    for i,isharpe in enumerate(sharpeList):
        if i == len(sharpeList)-1:
            if np.isnan(isharpe):
                sharpeList[i] = -999.
        else:
            if isharpe==np.nan:
                sharpeList[i] = 0.
    print("sharpeList = ", sharpeList)

    crossplot_rotationAngle = 33. * np.pi/180.
    for i,isharpe in enumerate(sharpeList):
        # combine sharpe ratios compouted over 2 different periods
        # - use an angle of 33 degrees instead of 45 to give slightly more weight the the "no gap" sharpe
        if i==0:
            continue
        elif i==1:
            sharpe_pair = [sharpeList[i-1],sharpeList[i]]
        else:
            sharpe_pair = [sharpe2periods,sharpeList[i]]
        sharpe2periods = sharpe_pair[0]*np.sin(crossplot_rotationAngle) + sharpe_pair[1]*np.cos(crossplot_rotationAngle)
        print("i, sharpe_pair, combined = ", i,sharpe_pair, sharpe2periods)

    return sharpe2periods

#----------------------------------------------

def recentTrendAndMidTrendChannelFitWithAndWithoutGap(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28,numdaysinfit2=20, offset=3):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    #recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.
    #residuals = x[-numdaysinfit-offset:-offset+1] - midTrend
    #fitStdDev = np.std(residuals)
    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    #print ".....gappedLowerFit, gappedUpperFit = ", gappedLowerFit, gappedUpperFit
    #print ".....fitStdDev,currentUpper,currentLower,x[-1] = ", fitStdDev, currentUpper,currentLower,x[-1]
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    numStdDevs = currentResidual / fitStdDev

    # calculate gain or loss over the period (with offset)
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)

    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit2,
                                           offset=0)
    #recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2+1,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    NoGapCurrentUpper = p(0) * 1.
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapCurrentLower = p(0) * 1.
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    return lowerTrend, upperTrend, NoGapLowerTrend, NoGapUpperTrend

#----------------------------------------------

def recentTrendAndMidTrendWithGap(x,datearray,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28,numdaysinfit2=20, offset=3):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.
    #residuals = x[-numdaysinfit-offset:-offset+1] - midTrend
    #fitStdDev = np.std(residuals)
    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    #print ".....gappedLowerFit, gappedUpperFit = ", gappedLowerFit, gappedUpperFit
    #print ".....fitStdDev,currentUpper,currentLower,x[-1] = ", fitStdDev, currentUpper,currentLower,x[-1]
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    numStdDevs = currentResidual / fitStdDev

    # calculate gain or loss over the period (with offset)
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)


    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit2,
                                           offset=0)
    recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    NoGapCurrentUpper = p(0) * 1.
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapCurrentLower = p(0) * 1.
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    # calculate relative gain or loss over entire period
    gainloss_cumu2 = NoGapMidTrend[-1]/midTrend[0] -1.
    relative_GainLossRatio = (NoGapCurrentUpper + NoGapCurrentLower)/(currentUpper + currentLower)

    import matplotlib.pylab as plt
    plt.figure(1)
    plt.clf()
    plt.grid(True)
    plt.plot(datearray[-(numdaysinfit+offset+20):],x[-(numdaysinfit+offset+20):],'k-')
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    plt.plot(datearray[np.array(relativedates)],upperTrend,'y-')
    plt.plot(datearray[np.array(relativedates)],lowerTrend,'y-')
    plt.plot([datearray[-1]],[(upperTrend[-1]+lowerTrend[-1])/2.],'y.',ms=30)
    relativedates = list(range(-numdaysinfit2,0))
    plt.plot(datearray[np.array(relativedates)],NoGapUpperTrend,'c-')
    plt.plot(datearray[np.array(relativedates)],NoGapLowerTrend,'c-')
    plt.plot([datearray[-1]],[(NoGapUpperTrend[-1]+NoGapLowerTrend[-1])/2.],'c.',ms=30)
    plt.show()

    return gainloss_cumu, gainloss_cumu2, numStdDevs, relative_GainLossRatio

#----------------------------------------------

def recentTrendComboGain(x,
                         datearray,
                         minperiod=4,
                         maxperiod=12,
                         incperiod=3,
                         numdaysinfit=28,
                         numdaysinfit2=20,
                         offset=3):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

    from scipy.stats import gmean

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    midTrend = (upperTrend+lowerTrend)/2.

    # calculate gain or loss over the period (no offset)
    gainloss_period = midTrend[1:] / midTrend[:-1]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = gmean( gainloss_period )**252 -1.

    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit2,
                                           offset=0)
    recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    # calculate gain or loss over the period (no offset)
    gainloss_period_nogap = NoGapMidTrend[1:] / NoGapMidTrend[:-1]
    gainloss_period_nogap[np.isnan(gainloss_period_nogap)] = 1.
    gainloss_cumu_nogap = gmean( gainloss_period_nogap )**252 -1.

    # calculate "combo gain" (defined as sum of gains rewarded for improvement, penalized for decline
    comboGain = (gainloss_cumu + gainloss_cumu_nogap)/2.
    comboGain *= (gainloss_cumu_nogap+1) / (gainloss_cumu+1)

    return comboGain

#----------------------------------------------

def textmessageOutsideTrendChannel(  symbols, adjClose ):

    # temporarily skip this!!!!!!
    #return

    import datetime

    from functions.SendEmail import SendTextMessage
    from functions.SendEmail import SendEmail

    # send text message for held stocks if the lastest quote is outside
    # (to downside) the established channel

    # Get Credentials for sending email
    params = GetParams()
    print("")

    #print "params = ", params
    print("")
    username = str(params['fromaddr']).split("@")[0]
    emailpassword = str(params['PW'])

    subjecttext = "PyTAAA update - Pct Trend Channel"
    boldtext = "time is "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    headlinetext = "market status: " + get_MarketOpenOrClosed()

    # Get Holdings from file
    holdings = GetHoldings()
    holdings_symbols = holdings['stocks']
    edition = GetEdition()

    # process symbols in current holdings
    downtrendSymbols = []
    channelPercent = []
    channelGainsLossesHoldings = []
    channelStdsHoldings = []
    channelGainsLosses = []
    channelStds = []
    currentNumStdDevs = []
    for i, symbol in enumerate(symbols):
        pctChannel,channelGainLoss,channelStd,numStdDevs = jumpTheChannelTest(adjClose[i,:],\
                                                                              #minperiod=4,\
                                                                              #maxperiod=12,\
                                                                              #incperiod=3,\
                                                                              #numdaysinfit=28,\
                                                                              #offset=3)
                                                   minperiod=params['minperiod'],
                                                   maxperiod=params['maxperiod'],
                                                   incperiod=params['incperiod'],
                                                   numdaysinfit=params['numdaysinfit'],
                                                   offset=params['offset'])
        channelGainsLosses.append(channelGainLoss)
        channelStds.append(channelStd)
        if symbol in holdings_symbols:
            #pctChannel = jumpTheChannelTest(adjClose[i,:],minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3)
            print(" ... performing PctChannelTest: symbol = ",format(symbol,'5s'), "  pctChannel = ", format(pctChannel-1.,'6.1%'))
            '''
            if pctChannel < 1.:
                # send textmessage alert of possible new down-trend
                downtrendSymbols.append(symbol)
                channelPercent.append(format(pctChannel-1.,'6.1%'))
            '''
            # send textmessage alert of current trend
            downtrendSymbols.append(symbol)
            channelPercent.append(format(pctChannel-1.,'6.1%'))
            channelGainsLossesHoldings.append(format(channelGainLoss,'6.1%'))
            channelStdsHoldings.append(format(channelStd,'6.1%'))
            currentNumStdDevs.append(format(numStdDevs,'6.1f'))

    print("\n ... downtrending symbols are ", downtrendSymbols, "\n")

    if len(downtrendSymbols) > 0:
        #--------------------------------------------------
        # send text message
        #--------------------------------------------------
        #text_message = "PyTAAA/"+edition+" shows "+str(downtrendSymbols)+" in possible downtrend... \n"+str(channelPercent)+" % of trend channel."
        text_message = "PyTAAA/"+edition+" shows "+str(downtrendSymbols)+" current trend... "+\
                       "\nPct of trend channel  = "+str(channelPercent)+\
                       "\nperiod gainloss     = "+str(channelGainsLossesHoldings)+\
                       "\nperiod gainloss std = "+str(channelStdsHoldings)+\
                       "\ncurrent # std devs  = "+str(currentNumStdDevs)

        print(text_message +"\n\n")

        # send text message if market is open
        if 'close in' in get_MarketOpenOrClosed():
            #SendTextMessage( username,emailpassword,params['toSMS'],params['fromaddr'],text_message )
            SendEmail(username,emailpassword,params['toSMS'],params['fromaddr'],subjecttext,text_message,boldtext,headlinetext)

    return


#----------------------------------------------

def SMA_2D(x,periods):
    SMA = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        minx = max(0,i-periods)
        SMA[:,i] = np.mean(x[:,minx:i+1],axis=-1)
    return SMA


#----------------------------------------------

def despike_2D(x,periods,stddevThreshold=5.0):
    # remove outliers from gradient of x (in 2nd dimension)
    gainloss = np.ones((x.shape[0],x.shape[1]),dtype=float)
    gainloss[:,1:] = x[:,1:] / x[:,:-1]
    for i in range( 1,x.shape[1] ):
        minx = max(0,i-periods)
        Stddev = np.std(gainloss[:,minx:i],axis=-1)
        Stddev *= stddevThreshold
        Stddev += 1.
        test = np.dstack( (Stddev, gainloss[:,i]) )
        gainloss[:,i] = np.min( test, axis=-1)
    gainloss[:,0] = x[:,0].copy()
    value = np.cumprod(gainloss,axis=1)
    return value


#----------------------------------------------

def SMA(x,periods):
    SMA = np.zeros( (x.shape[0]), dtype=float)
    for i in range( x.shape[0] ):
        minx = max(0,i-periods)
        SMA[i] = np.mean(x[minx:i+1],axis=-1)
    return SMA


#----------------------------------------------

def SMS(x,periods):
    _SMS = np.zeros( (x.shape[0]), dtype=float)
    for i in range( x.shape[0] ):
        minx = max(0,i-periods)
        _SMS[i] = np.sum(x[minx:i+1],axis=-1)
    return _SMS


#----------------------------------------------

def MoveMax_2D(x,periods):
    MMax = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        minx = max(0,i-periods)
        MMax[:,i] = np.max(x[:,minx:i+1],axis=-1)
    return MMax


#----------------------------------------------

def MoveMax(x,periods):
    MMax = np.zeros( (x.shape[0]), dtype=float)
    for i in range( x.shape[0] ):
        minx = max(0,i-periods)
        MMax[i] = np.max(x[minx:i+1],axis=-1)
    return MMax

#----------------------------------------------

def MoveMin(x,periods):
    MMin = np.zeros( (x.shape[0]), dtype=float)
    for i in range( x.shape[0] ):
        minx = max(0,i-periods)
        MMin[i] = np.min(x[minx:i+1],axis=-1)
    return MMin


#----------------------------------------------

def move_sharpe_2D(adjClose,dailygainloss,period):
    """
    Compute the moving sharpe ratio
      sharpe_ratio = ( gmean(PortfolioDailyGains[-lag:])**252 -1. )
                   / ( np.std(PortfolioDailyGains[-lag:])*sqrt(252) )
      formula assume 252 trading days per year

    Geometric mean is simplified as follows:
    where the geometric mean is being used to determine the average
    growth rate of some quantity, and the initial and final values
    of that quantity are known, the product of the measured growth
    rate at every step need not be taken. Instead, the geometric mean
    is simply ( a(n)/a(0) )**(1/n), where n is the number of steps
    """
    from scipy.stats import gmean
    from math import sqrt
    from numpy import std
    #
    sharpe = np.zeros( (adjClose.shape[0],adjClose.shape[1]), dtype=float)
    for i in range( dailygainloss.shape[1] ):
        minindex = max( i-period, 0 )
        if i > minindex :
            sharpe[:,i] = ( gmean(dailygainloss[:,minindex:i+1],axis=-1)**252 -1. )     \
                   / ( np.std(dailygainloss[:,minindex:i+1],axis=-1)*sqrt(252) )
        else :
            sharpe[:,i] = 0.

    sharpe[sharpe==0]=.05
    sharpe[isnan(sharpe)] =.05

    return sharpe


#----------------------------------------------

def computeSignal2D( adjClose, gainloss, params ):

    print(" ... inside computeSignal2D ... ")
    print(" params = ",params)
    MA1 = int(params['MA1'])
    MA2 = int(params['MA2'])
    MA2offset = int(params['MA2offset'])

    narrowDays = params['narrowDays']
    mediumDays = params['mediumDays']
    wideDays = params['wideDays']

    lowPct = float(params['lowPct'])
    hiPct = float(params['hiPct'])
    sma2factor = float(params['MA2factor'])
    uptrendSignalMethod = params['uptrendSignalMethod']

    if uptrendSignalMethod == 'SMAs' :
        print("  ...using 3 SMA's for signal2D")
        print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")
        ########################################################################
        ## Calculate signal for all stocks based on 3 simple moving averages (SMA's)
        ########################################################################
        sma0 = SMA_2D( adjClose, MA2 )               # MA2 is shortest
        sma1 = SMA_2D( adjClose, MA2 + MA2offset )
        sma2 = sma2factor * SMA_2D( adjClose, MA1 )  # MA1 is longest

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if adjClose[ii,jj] > sma2[ii,jj] or ((adjClose[ii,jj] > min(sma0[ii,jj],sma1[ii,jj]) and sma0[ii,jj] > sma0[ii,jj-1])):
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

        dailyNumberUptrendingStocks = np.sum(signal2D,axis = 0)

        return signal2D

    elif uptrendSignalMethod == 'minmaxChannels' :
        print("  ...using 3 minmax channels for signal2D")
        print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")

        ########################################################################
        ## Calculate signal for all stocks based on 3 minmax channels (dpgchannels)
        ########################################################################

        # narrow channel is designed to remove day-to-day variability

        print("narrow days min,max,inc = ", narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.)
        narrow_minChannel, narrow_maxChannel = dpgchannel_2D( adjClose, narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7. )
        narrow_midChannel = (narrow_minChannel+narrow_maxChannel)/2.

        medium_minChannel, medium_maxChannel = dpgchannel_2D( adjClose, mediumDays[0], mediumDays[-1], (mediumDays[-1]-mediumDays[0])/7. )
        medium_midChannel = (medium_minChannel+medium_maxChannel)/2.
        mediumSignal = ((narrow_midChannel-medium_minChannel)/(medium_maxChannel-medium_minChannel)-0.5)*2.0

        wide_minChannel, wide_maxChannel = dpgchannel_2D( adjClose, wideDays[0], wideDays[-1], (wideDays[-1]-wideDays[0])/7. )
        wide_midChannel = (wide_minChannel+wide_maxChannel)/2.
        wideSignal = ((narrow_midChannel-wide_minChannel)/(wide_maxChannel-wide_minChannel)-0.5)*2.0

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if mediumSignal[ii,jj] + wideSignal[ii,jj] > 0:
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

            '''
            # take care of special case where mp quote exists at end of series
            if firstTrailingEmptyPriceIndex[ii] != 0:
                signal2D[ii,firstTrailingEmptyPriceIndex[ii]:] = 0
            '''

        return signal2D

    elif uptrendSignalMethod == 'percentileChannels' :
        print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")
        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        lowChannel,hiChannel = percentileChannel_2D(adjClose,MA1,MA2+.01,MA2offset,lowPct,hiPct)
        for ii in range(adjClose.shape[0]):
            for jj in range(1,adjClose.shape[1]):
                if (adjClose[ii,jj] > lowChannel[ii,jj] and adjClose[ii,jj-1] <= lowChannel[ii,jj-1]) or adjClose[ii,jj] > hiChannel[ii,jj]:
                    signal2D[ii,jj] = 1
                elif (adjClose[ii,jj] < hiChannel[ii,jj] and adjClose[ii,jj-1] >= hiChannel[ii,jj-1]) or adjClose[ii,jj] < lowChannel[ii,jj]:
                    signal2D[ii,jj] = 0
                else:
                    signal2D[ii,jj] = signal2D[ii,jj-1]

                if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                    signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
            signal2D[ii,0:index] = 0

        print(" finished calculating signal2D... mean signal2D = ", signal2D.mean())

        return signal2D, lowChannel, hiChannel


#----------------------------------------------

def nanrms(x, axis=None):
    from bottleneck import nanmean
    return sqrt(nanmean(x**2, axis=axis))

#----------------------------------------------


def move_informationRatio(dailygainloss_portfolio,dailygainloss_index,period):
    """
    Compute the moving information ratio

      returns for stock (annualized) = Rs
        -- assuming 252 days per year this is gmean(dailyGains)**252 -1
        -- Rs denotes stock's return

      excess return compared to bendmark = Expectation(Rp - Ri)
        -- assuming 252 days per year this is sum(Rp - Ri)/252, or just mean(Rp - Ri)
        -- Rp denotes active portfolio return
        -- Ri denotes index return

      tracking error compared to bendmark = sqrt(Expectation((Rp - Ri)**2))
        -- assuming 252 days per year this is sqrt(Sum((Rp - Ri)**2)/252), or just sqrt(mean(((Rp - Ri)**2)))
        -- Rp denotes active portfolio return
        -- Ri denotes index return

      information_ratio = ExcessReturn / TrackingError

      formula assume 252 trading days per year

    Geometric mean is simplified as follows:
    where the geometric mean is being used to determine the average
    growth rate of some quantity, and the initial and final values
    of that quantity are known, the product of the measured growth
    rate at every step need not be taken. Instead, the geometric mean
    is simply ( a(n)/a(0) )**(1/n), where n is the number of steps
    """
    from scipy.stats import gmean
    from math import sqrt
    from numpy import std
    from bottleneck import nanmean
    #
    infoRatio = np.zeros( (dailygainloss_portfolio.shape[0],dailygainloss_portfolio.shape[1]), dtype=float)

    for i in range( dailygainloss_portfolio.shape[1] ):

        minindex = max( i-period, 0 )

        if i > minindex :
            returns_portfolio = dailygainloss_portfolio[:,minindex:i+1] -1.
            returns_index =  dailygainloss_index[minindex:i+1] -1.
            excessReturn = nanmean( returns_portfolio - returns_index, axis = -1 )
            trackingError = nanrms( dailygainloss_portfolio[:,minindex:i+1] - dailygainloss_index[minindex:i+1], axis = -1 )

            infoRatio[:,i] = excessReturn / trackingError

            if i == dailygainloss_portfolio.shape[1]-1:
                print(" returns_portfolio = ", returns_portfolio)
                print(" returns_index = ", returns_index)
                print(" excessReturn = ", excessReturn)
                print(" infoRatio[:,i] = ", infoRatio[:,i])

        else :
            infoRatio[:,i] *= 0.

    infoRatio[infoRatio==0]=.0
    infoRatio[isnan(infoRatio)] =.0

    return infoRatio

#----------------------------------------------

def multiSharpe( datearray, adjClose, periods ):

    maxPeriod = np.max( periods )

    dates = datearray[maxPeriod:]
    sharpesPeriod = np.zeros( (len(periods),len(dates)), 'float' )
    adjCloseSubset = adjClose[:,-len(dates):]

    for iperiod,period in enumerate(periods) :
        lenSharpe = period
        for idate in range( maxPeriod,adjClose.shape[1] ):
            sharpes = []
            for ii in range(adjClose.shape[0]):
                sharpes.append( allstats( adjClose[ii,idate-lenSharpe:idate] ).sharpe() )
            sharpes = np.array( sharpes )
            sharpes = sharpes[np.isfinite( sharpes )]
            if len(sharpes) > 0:
                sharpesAvg = np.mean(sharpes)
                if idate%1000 == 0:
                    print(period, datearray[idate],len(sharpes), sharpesAvg)
            else:
                sharpesAvg = 0.
            sharpesPeriod[iperiod,idate-maxPeriod] = sharpesAvg

    plotSharpe = sharpesPeriod[:,-len(dates):].copy()
    plotSharpe += .3
    plotSharpe /= 1.25
    signal = np.median(plotSharpe,axis=0)
    for i in range( plotSharpe.shape[0] ):
        signal += (np.clip( plotSharpe[i,:], -1., 2.) - signal)

    medianSharpe = np.median(plotSharpe,axis=0)
    signal = np.median(plotSharpe,axis=0) + 1.5 * (np.mean(plotSharpe,axis=0) - np.median(plotSharpe,axis=0))

    medianSharpe = np.clip( medianSharpe, -.1, 1.1 )
    signal = np.clip( signal, -.05, 1.05 )

    return dates, medianSharpe, signal


#----------------------------------------------

def move_martin_2D(adjClose,period):
    """
    Compute the moving martin ratio (ulcer performance index)

    martin ratio is based on ulcer index (rms drawdown over period)

    Reference: http://www.tangotools.com/ui/ui.htm


      martin_ratio = ( gmean(PortfolioDailyGains[-lag:])**252 -1. )
                   / ( np.std(PortfolioDailyGains[-lag:])*sqrt(252) )
      formula assume 252 trading days per year

    Geometric mean is simplified as follows:
    where the geometric mean is being used to determine the average
    growth rate of some quantity, and the initial and final values
    of that quantity are known, the product of the measured growth
    rate at every step need not be taken. Instead, the geometric mean
    is simply ( a(n)/a(0) )**(1/n), where n is the number of steps
    """
    from scipy.stats import gmean
    from math import sqrt
    from numpy import std
    #
    MoveMax = MoveMax_2D( adjClose, period )
    pctDrawDown = adjClose / MoveMax - 1.
    pctDrawDown = pctDrawDown ** 2

    martin = np.sqrt( SMA_2D( pctDrawDown, period )  )

    # reset NaN's to zero
    martin[ np.isnan(martin) ] = 0.

    return martin

#----------------------------------------------

def sharpeWeightedRank_2D(datearray,symbols,adjClose,signal2D,signal2D_daily,LongPeriod,rankthreshold,riskDownside_min,riskDownside_max,rankThresholdPct,stddevThreshold=4.,makeQCPlots=False):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    import nose
    import os
    import sys

    import matplotlib.gridspec as gridspec
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
        import scipy.stats.mstats as bn
    from .quotes_for_list_adjClose import get_Naz100List, get_SP500List


    # Get params for sending textmessage and email
    params = GetParams()
    stockList = params['stockList']

    adjClose_despike = despike_2D( adjClose, LongPeriod, stddevThreshold=stddevThreshold )

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[:,1:] = adjClose_despike[:,1:] / adjClose_despike[:,:-1]  ## experimental
    gainloss[isnan(gainloss)]=1.

    # convert signal2D to contain either 1 or 0 for weights
    signal2D -= signal2D.min()
    signal2D *= signal2D.max()

    # apply signal to daily gainloss
    print("\n\n\n######################\n...gainloss min,median,max = ",gainloss.min(),gainloss.mean(),np.median(gainloss),gainloss.max())
    print("...signal2D min,median,max = ",signal2D.min(),signal2D.mean(),np.median(signal2D),signal2D.max(),"\n\n\n")
    gainloss = gainloss * signal2D
    gainloss[gainloss == 0] = 1.0

    # update file with daily count of uptrending symbols in index universe
    filepath = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_dailyNumberUptrendingSymbolsList.txt" )
    print("\n\nfile for daily number of uptrending symbols = ", filepath)
    if os.path.exists( os.path.abspath(filepath) ):
        numberUptrendingSymbols = 0
        for i in range(len(symbols)):
            if signal2D_daily[i,-1] == 1.:
                numberUptrendingSymbols += 1
                #print "numberUptrendingSymbols,i,symbol,signal2D = ",numberUptrendingSymbols,i,symbols[i],signal2D_daily[i,-1]

        dailyUptrendingCount_text = "\n"+str(datearray[-1])+", "+str(numberUptrendingSymbols)
        with open( filepath, "a" ) as f:
            f.write(dailyUptrendingCount_text)
    else:
        dailyUptrendingCount_text = "date, daily count of uptrending symbols"
        with open( filepath, "w" ) as f:
            f.write(dailyUptrendingCount_text)
        numberUptrendingSymbols = np.zeros( signal2D_daily.shape[1], 'int' )
        for k in range(signal2D_daily.shape[1]):
            for i in range(len(symbols)):
                if signal2D_daily[i,k] == 1.:
                    numberUptrendingSymbols[k] += 1
                    #print "numberUptrendingSymbols,i,symbol,signal2D = ",numberUptrendingSymbols[k],datearray[k],symbols[i],signal2D_daily[i,k]

            dailyUptrendingCount_text = "\n"+str(datearray[k])+", "+str(numberUptrendingSymbols[k])
            with open( filepath, "a" ) as f:
                f.write(dailyUptrendingCount_text)

    value = 10000. * np.cumprod(gainloss,axis=1)

    # calculate gainloss over period of "LongPeriod" days
    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[:,LongPeriod:] = adjClose_despike[:,LongPeriod:] / adjClose_despike[:,:-LongPeriod]  ## experimental
    monthgainloss[isnan(monthgainloss)]=1.

    # apply signal to daily monthgainloss
    monthgainloss = monthgainloss * signal2D
    monthgainloss[monthgainloss == 0] = 1.0

    monthgainlossweight = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)

    rankweight = 1./rankthreshold


    ########################################################################
    ## Calculate change in rank of active stocks each day (without duplicates as ties)
    ########################################################################
    monthgainlossRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)
    monthgainlossPrevious = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainlossPreviousRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)

    monthgainlossRank = bn.rankdata(monthgainloss,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(monthgainlossRank)
    monthgainlossRank -= maxrank-1
    monthgainlossRank *= -1
    monthgainlossRank += 2

    monthgainlossPrevious[:,LongPeriod:] = monthgainloss[:,:-LongPeriod]
    monthgainlossPreviousRank = bn.rankdata(monthgainlossPrevious,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(monthgainlossPreviousRank)
    monthgainlossPreviousRank -= maxrank-1
    monthgainlossPreviousRank *= -1
    monthgainlossPreviousRank += 2

    # weight deltaRank for best and worst performers differently
    rankoffsetchoice = rankthreshold
    delta = -( monthgainlossRank.astype('float') - monthgainlossPreviousRank.astype('float') ) / ( monthgainlossRank.astype('float') + float(rankoffsetchoice) )

    # if rank is outside acceptable threshold, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
        for jj in range(monthgainloss.shape[1]):
            if monthgainloss[ii,jj] > rankThreshold :
                delta[ii,jj] = -monthgainloss.shape[0]/2
                if jj == monthgainloss.shape[1]:
                    print("*******setting delta (Rank) low... Stock has rank outside acceptable range... ",ii, symbols[ii], monthgainloss[ii,jj])


    # if symbol is not in current stock index universe, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    if stockList == 'Naz100':
        currentSymbolList,_,_ = get_Naz100List()
    elif stockList == 'SP500':
        currentSymbolList,_,_ = get_SP500List()
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
        if symbols[ii] not in currentSymbolList and symbols[ii] != 'CASH' :
            delta[ii,:] = -monthgainloss.shape[0]/2
            numisnans = adjClose[ii,:]
            # NaN in last value usually means the stock is removed from the index so is not updated, but history is still in HDF file
            print("*******setting delta (Rank) low... Stock is no longer in stock index universe... ",ii, symbols[ii])


    deltaRank = bn.rankdata( delta, axis=0 )

    # reverse the ranks (low deltaRank have the fastest improving rank)
    maxrank = np.max(deltaRank)
    deltaRank -= maxrank-1
    deltaRank *= -1
    deltaRank += 2

    for ii in range(monthgainloss.shape[1]):
        if deltaRank[:,ii].min() == deltaRank[:,ii].max():
            deltaRank[:,ii] = 0.

    ########################################################################
    ## Copy current day rankings deltaRankToday
    ########################################################################

    deltaRankToday = deltaRank[:,-1].copy()

    ########################################################################
    ## Hold values constant for calendar month (gains, ranks, deltaRanks)
    ########################################################################

    for ii in range(1,monthgainloss.shape[1]):
        if datearray[ii].month == datearray[ii-1].month:
            monthgainloss[:,ii] = monthgainloss[:,ii-1]
            delta[:,ii] = delta[:,ii-1]
            deltaRank[:,ii] = deltaRank[:,ii-1]

    ########################################################################
    ## Calculate number of active stocks each day
    ########################################################################

    # TODO: activeCount can be computed before loop to save CPU cycles
    # count number of unique values
    activeCount = np.zeros(adjClose.shape[1],dtype=float)
    for ii in np.arange(0,monthgainloss.shape[0]):
        firsttradedate = np.argmax( np.clip( np.abs( gainloss[ii,:]-1. ), 0., .00001 ) )
        activeCount[firsttradedate:] += 1

    minrank = np.min(deltaRank,axis=0)
    maxrank = np.max(deltaRank,axis=0)
    # convert rank threshold to equivalent percent of rank range

    rankthresholdpercentequiv = np.round(float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0])
    ranktest = deltaRank <= rankthresholdpercentequiv

    ########################################################################
    ### Calculate downside risk measure for weighting stocks.
    ### Use 1./ movingwindow_sharpe_ratio for risk measure.
    ### Modify weights with 1./riskDownside and scale so they sum to 1.0
    ########################################################################

    riskDownside = 1. / move_sharpe_2D(adjClose,gainloss,LongPeriod)
    riskDownside = np.clip( riskDownside, riskDownside_min, riskDownside_max)

    riskDownside[isnan(riskDownside)] = np.max(riskDownside[~isnan(riskDownside)])
    for ii in range(riskDownside.shape[0]) :
        riskDownside[ii] = riskDownside[ii] / np.sum(riskDownside,axis=0)

    ########################################################################
    ### calculate equal weights for ranks below threshold
    ########################################################################

    elsecount = 0
    elsedate  = 0
    for ii in np.arange(1,monthgainloss.shape[1]) :
        if activeCount[ii] > minrank[ii] and rankthresholdpercentequiv[ii] > 0:
            for jj in range(value.shape[0]):
                test = deltaRank[jj,ii] <= rankthresholdpercentequiv[ii]
                if test == True :
                    monthgainlossweight[jj,ii]  = 1./rankthresholdpercentequiv[ii]
                    monthgainlossweight[jj,ii]  = monthgainlossweight[jj,ii] / riskDownside[jj,ii]
                else:
                    monthgainlossweight[jj,ii]  = 0.
        elif activeCount[ii] == 0 :
            monthgainlossweight[:,ii]  *= 0.
            monthgainlossweight[:,ii]  += 1./adjClose.shape[0]
        else :
            elsedate = datearray[ii]
            elsecount += 1
            monthgainlossweight[:,ii]  = 1./activeCount[ii]

    aaa = np.sum(monthgainlossweight,axis=0)


    allzerotest = np.sum(monthgainlossweight,axis=0)
    sumallzerotest = allzerotest[allzerotest == 0].shape
    if sumallzerotest > 0:
        print("")
        print(" invoking correction to monthgainlossweight.....")
        print("")
        for ii in np.arange(1,monthgainloss.shape[1]) :
            if np.sum(monthgainlossweight[:,ii]) == 0:
                monthgainlossweight[:,ii]  = 1./activeCount[ii]

    print(" weights calculation else clause encountered :",elsecount," times. last date encountered is ",elsedate)
    rankweightsum = np.sum(monthgainlossweight,axis=0)

    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    monthgainlossweight = monthgainlossweight / np.sum(monthgainlossweight,axis=0)
    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    if makeQCPlots==True:
        # input symbols and company names from text file
        if stockList == 'Naz100':
            companyName_file = os.path.join( os.getcwd(), "symbols",  "companyNames.txt" )
        elif stockList == 'SP500':
            companyName_file = os.path.join( os.getcwd(), "symbols",  "SP500_companyNames.txt" )
        with open( companyName_file, "r" ) as f:
            companyNames = f.read()

        print("\n\n\n")
        companyNames = companyNames.split("\n")
        ii = companyNames.index("")
        del companyNames[ii]
        companySymbolList  = []
        companyNameList = []
        for iname,name in enumerate(companyNames):
            name = name.replace("amp;", "")
            testsymbol, testcompanyName = name.split(";")
            companySymbolList.append(testsymbol)
            companyNameList.append(testcompanyName)

        # print list showing current rankings and weights
        # - symbol
        # - rank (at begining of month)
        # - rank (most recent trading day)
        # - weight from sharpe ratio
        # - price
        import os

        rank_text = "<div id='rank_table_container'><h3>"+"<p>Current stocks, with ranks, weights, and prices are :</p></h3><font face='courier new' size=3><table border='1'> \
                   <tr><td>Rank (start of month) \
                   </td><td>Rank (today) \
                   </td><td>Symbol \
                   </td><td>Company \
                   </td><td>Weight \
                   </td><td>Price  \
                   </td><td>Trend  \
                   </td><td>recent Gain or Loss (excludes a few days)  \
                   </td><td>stdDevs above or below trend  \
                   </td><td>trends ratio (%) with & wo gap  \
                   </td><td>P/E ratio \
                   </td></tr>\n"
        ChannelPct_text = "channelPercent:"
        channelPercent = []
        channelGainsLosses = []
        channelComboGainsLosses = []
        stdevsAboveChannel = []
        floatChannelGainsLosses = []
        floatChannelComboGainsLosses = []
        floatStdevsAboveChannel = []
        trendsRatio = []
        sharpeRatio = []
        floatSharpeRatio = []
        for i, isymbol in enumerate(symbols):
            ### save current projected position in price channel calculated without recent prices
            channelGainLoss, numStdDevs, pctChannel = recentTrendAndStdDevs(adjClose[i,:],
                                                              datearray,
                                                              minperiod=params['minperiod'],
                                                              maxperiod=params['maxperiod'],
                                                              incperiod=params['incperiod'],
                                                              numdaysinfit=params['numdaysinfit'],
                                                              offset=params['offset'])

            print("\nsymbol = ", symbols[i])
            sharpe2periods = recentSharpeWithAndWithoutGap(adjClose[i,:])

            print(" ... performing PctChannelTest: symbol = ",format(isymbol,'5s'), "  numStdDevs = ", format(numStdDevs,'6.1f'))
            channelGainsLosses.append(format(channelGainLoss,'6.1%'))
            stdevsAboveChannel.append(format(numStdDevs,'6.1f'))
            floatChannelGainsLosses.append(channelGainLoss)
            floatStdevsAboveChannel.append(numStdDevs)
            ChannelPct_text = ChannelPct_text + format(pctChannel-1.,'6.1%')
            sharpeRatio.append(format(sharpe2periods,'6.1f'))
            floatSharpeRatio.append(sharpe2periods)
            print("isymbol,floatSharpeRatio = ", isymbol,floatSharpeRatio[-1])

            channelComboGainLoss = recentTrendComboGain(adjClose[i,:],
                                                              datearray,
                                                              minperiod=params['minperiod'],
                                                              maxperiod=params['maxperiod'],
                                                              incperiod=params['incperiod'],
                                                              numdaysinfit=params['numdaysinfit'],
                                                              offset=params['offset'])

            #print " companyName, channelComboGainLoss = ", companyNameList[i], channelComboGainLoss
            channelComboGainsLosses.append(format(channelComboGainLoss,'6.1%'))
            floatChannelComboGainsLosses.append(channelComboGainLoss)

            lowerTrend, upperTrend, NoGapLowerTrend, NoGapUpperTrend = \
                     recentTrendAndMidTrendChannelFitWithAndWithoutGap( \
                                   adjClose[i,:], \
                                   minperiod=params['minperiod'], \
                                   maxperiod=params['maxperiod'], \
                                   incperiod=params['incperiod'], \
                                   numdaysinfit=params['numdaysinfit'], \
                                   numdaysinfit2=params['numdaysinfit2'], \
                                   offset=params['offset'])
            midTrendEndPoint = (lowerTrend[-1]+upperTrend[-1])/2.
            noGapMidTrendEndPoint = (NoGapLowerTrend[-1]+NoGapUpperTrend[-1])/2.
            trendsRatio.append( noGapMidTrendEndPoint/midTrendEndPoint - 1. )

        path_symbolChartsSort_byRankBeginMonth = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_symbolCharts_MonthStartRank.html" )
        path_symbolChartsSort_byRankToday = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_symbolCharts_TodayRank.html" )
        path_symbolChartsSort_byRecentGainRank = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_symbolCharts_recentGainRank.html" )
        path_symbolChartsSort_byRecentComboGainRank = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_symbolCharts_recentComboGainRank.html" )
        path_symbolChartsSort_byRecentTrendsRatioRank = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_symbolCharts_recentTrendRatioRank.html" )
        path_symbolChartsSort_byRecentSharpeRatioRank = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_symbolCharts_recentSharpeRatioRank.html" )

        pagetext_byRankBeginMonth = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Ranking at Start of Month</h1>+\n"
        pagetext_byRankToday = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Ranking Today</h1>+\n"
        pagetext_byRecentGainRank = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Recent Gain Ranking</h1>+\n"
        pagetext_byRecentComboGainRank = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Recent Combo Gain Ranking</h1>+\n"
        pagetext_byRecentTrendRatioRank = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Recent Trend Ratio Ranking</h1>+\n"
        pagetext_byRecentSharpeRatioRank = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Recent Sharpe Ratio Ranking</h1>+\n"

        floatChannelGainsLosses = np.array(floatChannelGainsLosses)
        floatChannelGainsLosses[np.isinf(floatChannelGainsLosses)] = -999.
        floatChannelGainsLosses[np.isneginf(floatChannelGainsLosses)] = -999.
        floatChannelGainsLosses[np.isnan(floatChannelGainsLosses)] = -999.
        floatChannelComboGainsLosses = np.array(floatChannelComboGainsLosses)
        floatChannelComboGainsLosses[np.isinf(floatChannelComboGainsLosses)] = -999.
        floatChannelComboGainsLosses[np.isneginf(floatChannelComboGainsLosses)] = -999.
        floatChannelComboGainsLosses[np.isnan(floatChannelComboGainsLosses)] = -999.
        floatStdevsAboveChannel = np.array(floatStdevsAboveChannel)
        floatStdevsAboveChannel[np.isinf(floatStdevsAboveChannel)] = -999.
        floatStdevsAboveChannel[np.isneginf(floatStdevsAboveChannel)] = -999.
        floatStdevsAboveChannel[np.isnan(floatStdevsAboveChannel)] = -999.
        floatTrendsRatio = np.array(trendsRatio)
        floatTrendsRatio[np.isinf(floatTrendsRatio)] = -999.
        floatTrendsRatio[np.isneginf(floatTrendsRatio)] = -999.
        floatTrendsRatio[np.isnan(floatTrendsRatio)] = -999.
        floatSharpeRatio = np.array(floatSharpeRatio)
        floatSharpeRatio[np.isinf(floatSharpeRatio)] = -999.
        floatSharpeRatio[np.isneginf(floatSharpeRatio)] = -999.
        floatSharpeRatio[np.isnan(floatSharpeRatio)] = -999.

        RecentGainRank = len(floatChannelGainsLosses) - bn.rankdata( floatChannelGainsLosses )
        RecentComboGainRank = len(floatChannelComboGainsLosses) - bn.rankdata( floatChannelComboGainsLosses )
        RecentGainStdDevRank = len(floatStdevsAboveChannel)- bn.rankdata( floatStdevsAboveChannel )
        RecentOrder = np.argsort( RecentGainRank + RecentGainStdDevRank )
        RecentRank = np.argsort( RecentOrder )
        RecentTrendsRatioRank = len(floatTrendsRatio) - bn.rankdata( floatTrendsRatio )
        RecentSharpeRatioRank = len(floatSharpeRatio) - bn.rankdata( floatSharpeRatio )

        peList = []
        floatPE_list = []
        for i, isymbol in enumerate(symbols):
            try:
                pe = getQuote(isymbol)['PE'][0]
            except:
                try:
                    pe = getQuote(isymbol)['PE'][0]
                except:
                    pe = np.nan
            floatPE_list.append(pe)
            peList.append(str(pe))

        for i, isymbol in enumerate(symbols):
            for j in range(len(symbols)):

                if int( deltaRank[j,-1] ) == i :
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'

                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(symbols[j])
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""

                    pe = peList[j]
                    rank_text = rank_text + \
                           "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                           "<td>" + format(deltaRankToday[j],'6.0f')  + \
                           "<td>" + format(symbols[j],'5s')  + \
                           "<td>" + format(companyName,'15s')  + \
                           "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                           "<td>" + format(adjClose[j,-1],'6.2f')  + \
                           "<td>" + trend  + \
                           "<td>" + channelGainsLosses[j]  + \
                           "<td>" + stdevsAboveChannel[j]  + \
                           "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                           "<td>" + pe  + \
                           "</td></tr>  \n"

                    ###print " i,j,companyName = ", i,j,"__"+companyName+"__"
                    if companyName != "":
                        if i==1:
                            avgChannelGainsLosses = floatChannelGainsLosses[j]
                            avgStdevsAboveChannel = floatStdevsAboveChannel[j]
                        else:
                            avgChannelGainsLosses = (avgChannelGainsLosses*(i-1)+floatChannelGainsLosses[j])/(i)
                            avgStdevsAboveChannel = (avgStdevsAboveChannel*(i-1)+floatStdevsAboveChannel[j])/(i)


                if i == deltaRank[j,-1]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(symbols[j])
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRankBeginMonth = pagetext_byRankBeginMonth +"<br><p> </p><p> </p><p> </p>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == deltaRankToday[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(symbols[j])
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRankToday = pagetext_byRankToday +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank <br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == RecentRank[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(symbols[j])
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRecentGainRank = pagetext_byRecentGainRank +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == RecentComboGainRank[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(symbols[j])
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRecentComboGainRank = pagetext_byRecentComboGainRank +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Combo Gain or Loss<br>(w and wo gap)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelComboGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == RecentTrendsRatioRank[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(symbols[j])
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRecentTrendRatioRank = pagetext_byRecentTrendRatioRank +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == RecentSharpeRatioRank[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(symbols[j])
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRecentSharpeRatioRank = pagetext_byRecentSharpeRatioRank +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Sharpe"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>sharpe ratio (%)<br>multiple periods"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.0f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatSharpeRatio[j],'5.1f') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

        medianChannelGainsLosses = np.median(floatChannelGainsLosses)
        medianTrendsRatio = np.median(floatTrendsRatio)
        avgTrendsRatio = np.mean(floatTrendsRatio)
        medianStdevsAboveChannel = np.median(floatStdevsAboveChannel)

        print("peList = ", floatPE_list)
        floatPE_list = np.array(floatPE_list)
        floatPE_list = floatPE_list[~np.isinf(floatPE_list)]
        floatPE_list = floatPE_list[~np.isneginf(floatPE_list)]
        floatPE_list = floatPE_list[~np.isnan(floatPE_list)]
        averagePE = np.mean(floatPE_list)
        medianPE = np.median(floatPE_list)

        avg_performance_text = "\n\n\n<font face='courier new' size=5><p>Average recent performance:</p></h3><font face='courier new' size=4>"+\
                               "<p>average trend excluding several days  = "+format(avgChannelGainsLosses,'6.1%')+"<br>"+\
                               "median trend excluding several days  = "+format(medianChannelGainsLosses,'6.1%')+"</p></h3><font face='courier new' size=4>"+\
                               "<p>average ratio of trend wo & with last several days  = "+format(avgTrendsRatio,'6.1%')+"<br>"+\
                               "median ratio of trend wo & with last several days  = "+format(medianTrendsRatio,'6.1%')+"</p></h3><font face='courier new' size=4>"+\
                               "<p>average number stds above/below trend = "+format(avgStdevsAboveChannel,'5.1f')+"<br>"+\
                               "median number stds above/below trend = "+format(medianStdevsAboveChannel,'5.1f')+"</p></h3><font face='courier new' size=4>"+\
                               "<p>average P/E = "+format(averagePE,'5.1f')+"<br>"+\
                               "median P/E = "+format(medianPE,'5.1f')+"</p></h3><font face='courier new' size=4>\n\n"

        rank_text = avg_performance_text + rank_text + "</table></div>\n"

        filepath = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_RankList.txt" )
        with open( filepath, "w" ) as f:
            f.write(rank_text)

        filepath = path_symbolChartsSort_byRankBeginMonth
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRankBeginMonth)

        filepath = path_symbolChartsSort_byRankToday
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRankToday)

        filepath = path_symbolChartsSort_byRecentGainRank
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRecentGainRank)

        filepath = path_symbolChartsSort_byRecentComboGainRank
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRecentComboGainRank)

        filepath = path_symbolChartsSort_byRecentTrendsRatioRank
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRecentTrendRatioRank)

        filepath = path_symbolChartsSort_byRecentSharpeRatioRank
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRecentSharpeRatioRank)

        ########################################################################
        ### save current ranks to params file
        ########################################################################

        lastdate_text = "lastdate: " + str(datearray[-1])
        symbol_text = "symbols: "
        rank_text = "ranks:"
        #####ChannelPct_text = "channelPercent:"
        """
        for i, isymbol in enumerate(symbols):
            symbol_text = symbol_text + format(symbols[i],'6s')
            rank_text = rank_text + format(deltaRankToday[i],'6.0f')
        """

        for i, isymbol in enumerate(symbols):
            for j in range(len(symbols)):
                if int( deltaRank[j,-1] ) == i :
                    symbol_text = symbol_text + format(symbols[j],'6s')
                    rank_text = rank_text + format(deltaRankToday[j],'6.0f')

            #####pctChannel = jumpTheChannelTest(adjClose[i,:],minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3)
            #####print " ... performing PctChannelTest: symbol = ",format(symbol,'5s'), "  pctChannel = ", format(pctChannel-1.,'6.1%')
            #####channelPercent.append(format(pctChannel-1.,'6.1%'))
            #####ChannelPct_text = ChannelPct_text + format(pctChannel-1.,'6.1%')

        filepath = os.path.join( os.getcwd(), "PyTAAA_ranks.params" )
        with open( filepath, "a" ) as f:
            f.write(lastdate_text)
            f.write("\n")
            f.write(symbol_text)
            f.write("\n")
            f.write(rank_text)
            f.write("\n")
            f.write(ChannelPct_text)
            f.write("\n")
        print("leaving function sharpeWeightedRank_2D...")

    return monthgainlossweight

#----------------------------------------------

def MAA_WeightedRank_2D(datearray,symbols,adjClose,signal2D,signal2D_daily,LongPeriod,numberStocksTraded,
                        wR, wC, wV, wS, stddevThreshold=4. ):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    import nose
    import os
    import sys

    import matplotlib.gridspec as gridspec
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
        import scipy.stats.mstats as bn


    params = GetParams()
    stockList = params['stockList']

    adjClose_despike = despike_2D( adjClose, LongPeriod, stddevThreshold=stddevThreshold )

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[:,1:] = adjClose_despike[:,1:] / adjClose_despike[:,:-1]  ## experimental
    gainloss[isnan(gainloss)]=1.

    # convert signal2D to contain either 1 or 0 for weights
    signal2D -= signal2D.min()
    signal2D *= signal2D.max()

    ############################
    ###
    ### filter universe of stocks to exclude all that have return < 0
    ### - needed for correlation to "equal weight index" (EWI)
    ### - EWI is daily gain/loss percentage
    ###
    ############################

    EWI  = np.zeros( adjClose.shape[1], 'float' )
    EWI_count  = np.zeros( adjClose.shape[1], 'int' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            if signal2D_daily[ii,jj] == 1:
                EWI[jj] += gainloss[ii,jj]
                EWI_count[jj] += 1
    EWI = EWI/EWI_count
    EWI[np.isnan(EWI)] = 1.0

    ############################
    ###
    ### compute correlation to EWI
    ### - each day, for each stock
    ### - not needed for stocks on days with return < 0
    ###
    ############################

    corrEWI  = np.zeros( adjClose.shape, 'float' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            start_date = max( jj - LongPeriod, 0 )
            if adjClose_despike[ii,jj] > adjClose_despike[ii,start_date]:
                corrEWI[ii,jj] = normcorrcoef(gainloss[ii,start_date:jj]-1.,EWI[start_date:jj]-1.)
                if corrEWI[ii,jj] <0:
                    corrEWI[ii,jj] = 0.

    ############################
    ###
    ### compute weights
    ### - each day, for each stock
    ### - set to 0. for stocks on days with return < 0
    ###
    ############################

    weights  = np.zeros( adjClose.shape, 'float' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            start_date = max( jj - LongPeriod, 0 )
            returnForPeriod = (adjClose_despike[ii,jj]/adjClose_despike[ii,start_date])-1.
            if returnForPeriod  < 0.:
                returnForPeriod = 0.
            volatility = np.std(adjClose_despike[ii,start_date:jj])
            weights[ii,jj] = ( returnForPeriod**wR * (1.-corrEWI[ii,jj])**wC / volatility**wV ) **wS

    weights[np.isnan(weights)] = 0.0

    # make duplicate of weights for adjusting using crashProtection
    CPweights = weights.copy()
    CP_cashWeight = np.zeros(adjClose.shape[1], 'float' )
    for jj in np.arange(adjClose.shape[1]) :
        weightsToday = weights[:,jj]
        CP_cashWeight[jj] = float(len(weightsToday[weightsToday==0.])) / len(weightsToday)

    ############################
    ###
    ### compute weights ranking and keep best
    ### 'best' are numberStocksTraded*%risingStocks
    ### - weights need to sum to 100%
    ###
    ############################

    weightRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)

    weightRank = bn.rankdata(weights,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(weightRank)
    weightRank -= maxrank-1
    weightRank *= -1
    weightRank += 2

    # set top 'numberStocksTraded' to have weights sum to 1.0
    for jj in np.arange(adjClose.shape[1]) :
        ranksToday = weightRank[:,jj].copy()
        weightsToday = weights[:,jj].copy()
        weightsToday[ranksToday > numberStocksTraded] = 0.
        if np.sum(weightsToday) > 0.:
            weights[:,jj] = weightsToday / np.sum(weightsToday)
        else:
            weights[:,jj] = 1./len(weightsToday)

    # set CASH to have weight based on CrashProtection
    cash_index = symbols.index("CASH")
    for jj in np.arange(adjClose.shape[1]) :
        CPweights[ii,jj] = CP_cashWeight[jj]
        weightRank[ii,jj] = 0
        ranksToday = weightRank[:,jj].copy()
        weightsToday = CPweights[:,jj].copy()
        weightsToday[ranksToday > numberStocksTraded] = 0.
        if np.sum(weightsToday) > 0.:
            CPweights[:,jj] = weightsToday / np.sum(weightsToday)
        else:
            CPweights[:,jj] = 1./len(weightsToday)

    # hold weights constant for month
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        start_date = max( jj - LongPeriod, 0 )
        yesterdayMonth = datearray[jj-1].month
        todayMonth = datearray[jj].month
        if todayMonth == yesterdayMonth:
            weights[:,jj] = weights[:,jj-1]
            CPweights[:,jj] = CPweights[:,jj-1]

    # input symbols and company names from text file
    if stockList == 'Naz100':
        companyName_file = os.path.join( os.getcwd(), "symbols",  "companyNames.txt" )
    elif stockList == 'SP500':
        companyName_file = os.path.join( os.getcwd(), "symbols",  "SP500_companyNames.txt" )
    with open( companyName_file, "r" ) as f:
        companyNames = f.read()

    print("\n\n\n")
    companyNames = companyNames.split("\n")
    ii = companyNames.index("")
    del companyNames[ii]
    companySymbolList  = []
    companyNameList = []
    for iname,name in enumerate(companyNames):
        name = name.replace("amp;", "")
        testsymbol, testcompanyName = name.split(";")
        companySymbolList.append(testsymbol)
        companyNameList.append(testcompanyName)

    # print list showing current rankings and weights
    # - symbol
    # - rank (at begining of month)
    # - rank (most recent trading day)
    # - weight from sharpe ratio
    # - price
    import os
    rank_text = "<div id='rank_table_container'><h3>"+"<p>Current stocks, with ranks, weights, and prices are :</p></h3><font face='courier new' size=3><table border='1'> \
               </td><td>Rank (today) \
               </td><td>Symbol \
               </td><td>Company \
               </td><td>Weight \
               </td><td>CP Weight \
               </td><td>Price  \
               </td><td>Trend  \
               </td></tr>\n"
    for i, isymbol in enumerate(symbols):
        for j in range(len(symbols)):
            if int( weightRank[j,-1] ) == i :
                if signal2D_daily[j,-1] == 1.:
                    trend = 'up'
                else:
                    trend = 'down'

                # search for company name
                try:
                    symbolIndex = companySymbolList.index(symbols[j])
                    companyName = companyNameList[symbolIndex]
                except:
                    companyName = ""

                rank_text = rank_text + \
                       "<tr><td>" + format(weightRank[j,-1],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(weights[j,-1],'5.03f') + \
                       "<td>" + format(CPweights[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "</td></tr>  \n"
    rank_text = rank_text + "</table></div>\n"

    print("leaving function MAA_WeightedRank_2D...")

    """
    print " symbols = ", symbols
    print " weights = ", weights[:,-1]
    print " CPweights = ", CPweights[:,-1]

    print " number NaNs in weights = ", weights[np.isnan(weights)].shape
    print " number NaNs in CPweights = ", CPweights[np.isnan(CPweights)].shape

    print " NaNs in monthgainlossweight = ", weights[np.isnan(weights)].shape
    testsum = np.sum(weights,axis=0)
    print " testsum shape, min, and max = ", testsum.shape, testsum.min(), testsum.max()
    """

    return weights, CPweights


#----------------------------------------------
def UnWeightedRank_2D(datearray,adjClose,signal2D,LongPeriod,rankthreshold,riskDownside_min,riskDownside_max,rankThresholdPct):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    import nose
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
        import scipy.stats.mstats as bn


    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.

    # convert signal2D to contain either 1 or 0 for weights
    signal2D -= signal2D.min()
    signal2D *= signal2D.max()

    # apply signal to daily gainloss
    gainloss = gainloss * signal2D
    gainloss[gainloss == 0] = 1.0

    value = 10000. * np.cumprod(gainloss,axis=1)

    # calculate gainloss over period of "LongPeriod" days
    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[isnan(monthgainloss)]=1.

    monthgainlossweight = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)

    rankweight = 1./rankthreshold

    ########################################################################
    ## Calculate change in rank of active stocks each day (without duplicates as ties)
    ########################################################################
    monthgainlossRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)
    monthgainlossPrevious = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainlossPreviousRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)

    monthgainlossRank = bn.rankdata(monthgainloss,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(monthgainlossRank)
    monthgainlossRank -= maxrank-1
    monthgainlossRank *= -1
    monthgainlossRank += 2

    monthgainlossPrevious[:,LongPeriod:] = monthgainloss[:,:-LongPeriod]
    monthgainlossPreviousRank = bn.rankdata(monthgainlossPrevious,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(monthgainlossPreviousRank)
    monthgainlossPreviousRank -= maxrank-1
    monthgainlossPreviousRank *= -1
    monthgainlossPreviousRank += 2

    # weight deltaRank for best and worst performers differently
    rankoffsetchoice = rankthreshold
    delta = -(monthgainlossRank - monthgainlossPreviousRank ) / (monthgainlossRank + rankoffsetchoice)

    # if rank is outside acceptable threshold, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
        for jj in range(monthgainloss.shape[1]):
            if monthgainloss[ii,jj] > rankThreshold :
                delta[ii,jj] = -monthgainloss.shape[0]/2

    deltaRank = bn.rankdata(delta,axis=0)
    # reverse the ranks (low deltaRank have the fastest improving rank)
    maxrank = np.max(deltaRank)
    deltaRank -= maxrank-1
    deltaRank *= -1
    deltaRank += 2

    for ii in range(monthgainloss.shape[1]):
        if deltaRank[:,ii].min() == deltaRank[:,ii].max():
            deltaRank[:,ii] = 0.

    ########################################################################
    ## Hold values constant for calendar month (gains, ranks, deltaRanks)
    ########################################################################

    for ii in np.arange(1,monthgainloss.shape[1]):
        if datearray[ii].month == datearray[ii-1].month:
            monthgainloss[:,ii] = monthgainloss[:,ii-1]
            deltaRank[:,ii] = deltaRank[:,ii-1]

    ########################################################################
    ## Calculate number of active stocks each day
    ########################################################################

    # TODO: activeCount can be computed before loop to save CPU cycles
    # count number of unique values
    activeCount = np.zeros(adjClose.shape[1],dtype=float)
    for ii in np.arange(0,monthgainloss.shape[0]):
        firsttradedate = np.argmax( np.clip( np.abs( gainloss[ii,:]-1. ), 0., .00001 ) )
        activeCount[firsttradedate:] += 1

    minrank = np.min(deltaRank,axis=0)
    maxrank = np.max(deltaRank,axis=0)
    # convert rank threshold to equivalent percent of rank range

    rankthresholdpercentequiv = np.round(float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0])
    ranktest = deltaRank <= rankthresholdpercentequiv

    ########################################################################
    ### calculate equal weights for ranks below threshold
    ########################################################################

    elsecount = 0
    elsedate  = 0
    for ii in np.arange(1,monthgainloss.shape[1]) :
        if activeCount[ii] > minrank[ii] and rankthresholdpercentequiv[ii] > 0:
            for jj in range(value.shape[0]):
                test = deltaRank[jj,ii] <= rankthresholdpercentequiv[ii]
                if test == True :
                    monthgainlossweight[jj,ii]  = 1./rankthresholdpercentequiv[ii]
                else:
                    monthgainlossweight[jj,ii]  = 0.
        elif activeCount[ii] == 0 :
            monthgainlossweight[:,ii]  *= 0.
            monthgainlossweight[:,ii]  += 1./adjClose.shape[0]
        else :
            elsedate = datearray[ii]
            elsecount += 1
            monthgainlossweight[:,ii]  = 1./activeCount[ii]

    aaa = np.sum(monthgainlossweight,axis=0)

    print("")
    print(" invoking correction to monthgainlossweight.....")
    print("")
    # find first date with number of stocks trading (rankthreshold) + 2
    activeCountAboveMinimum = activeCount
    activeCountAboveMinimum += -rankthreshold + 2
    firstTradeDate = np.argmax( np.clip( activeCountAboveMinimum, 0 , 1 ) )
    for ii in np.arange(firstTradeDate,monthgainloss.shape[1]) :
        if np.sum(monthgainlossweight[:,ii]) == 0:
            for kk in range(rankthreshold):
                indexHighDeltaRank = np.argmin(deltaRank[:,ii]) # remember that best performance is lowest deltaRank
                monthgainlossweight[indexHighDeltaRank,ii]  = 1./rankthreshold
                deltaRank[indexHighDeltaRank,ii] = 1000.


    print(" weights calculation else clause encountered :",elsecount," times. last date encountered is ",elsedate)
    rankweightsum = np.sum(monthgainlossweight,axis=0)

    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    monthgainlossweight = monthgainlossweight / np.sum(monthgainlossweight,axis=0)
    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    return monthgainlossweight








def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse.

    Parameters
    ----------
    X
        list
        a time series

    Returns
    -------
    H
        float
        Hurst exponent

    Examples
    --------
    >>> import pyeeg
    >>> from numpy.random import randn
    >>> a = randn(4096)
    >>> pyeeg.hurst(a)
    >>> 0.5057444

    ######################## Function contributed by Xin Liu #################
    https://code.google.com/p/pyeeg/source/browse/pyeeg.py
    Copyleft 2010 Forrest Sheng Bao http://fsbao.net
    PyEEG, a Python module to extract EEG features, v 0.02_r2
    Project homepage: http://pyeeg.org

    **Naming convention**

    Constants: UPPER_CASE_WITH_UNDERSCORES, e.g., SAMPLING_RATE, LENGTH_SIGNAL.
    Function names: lower_case_with_underscores, e.g., spectrum_entropy.
    Variables (global and local): CapitalizedWords or CapWords, e.g., Power.
    If a variable name consists of one letter, I may use lower case, e.g., x, y.

    """

    from numpy import zeros, log, array, cumsum, std
    from numpy.linalg import lstsq

    N = len(X)

    T = array([float(i) for i in range(1,N+1)])
    Y = cumsum(X)
    Ave_T = Y/T

    S_T = zeros((N))
    R_T = zeros((N))
    for i in range(N):
        S_T[i] = std(X[:i+1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = log(R_S)
    n = log(T).reshape(N, 1)
    H = lstsq(n[1:], R_S[1:])[0]
    return H[0]
