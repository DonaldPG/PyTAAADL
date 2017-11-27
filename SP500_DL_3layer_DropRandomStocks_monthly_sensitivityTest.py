###
### A Multilayer Perceptron implementation example using TensorFlow library.
###

import os
import time
from time import sleep
import datetime
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from matplotlib import pyplot as plt

#import pandas as pd
#from matplotlib import pylab as plt
#plt.ion()

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
                                  generatePredictionInput3layer, \
                                  interpolate, \
                                  cleantobeginning, \
                                  cleantoend
from functions.UpdateSymbols_inHDF5 import UpdateHDF5, \
                                           loadQuotes_fromHDF
os.chdir(_cwd)


# --------------------------------------------------
# Import list of symbols to process.
# --------------------------------------------------

# read list of symbols from disk.
stockList = 'Naz100'
#stockList = 'SP500'
#stockList = 'SP_wo_Naz'
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

if stockList == 'SP_wo_Naz':
    naz_filename = os.path.join(_data_path, 'symbols', 'Naz100_Symbols.txt')
    _, naz_symbols, _, _, _ = loadQuotes_fromHDF(naz_filename)
    ncount = 0
    scount = 0
    for i in range(len(symbols)-1, -1, -1):
        ticker = symbols[i]
        if ticker not in naz_symbols:
            scount += 1
            print(i, scount, ticker, adjClose[i, -1])
        else:
            ncount += 1
            print("...", ncount, ticker)
            adjClose = np.delete(adjClose, (i), axis=0)
            symbols.remove(ticker)

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

gainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
activeCount = np.zeros(adjClose.shape[1], dtype=float)

numberSharesCalc = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
gainloss[np.isnan(gainloss)] = 1.
value = 10000. * np.cumprod(gainloss, axis=1)   ### used bigger constant for inverse of quotes
BuyHold = np.average(value, axis=0)
BuyHoldFinalValue = np.average(value, axis=0)[-1]

print(" gainloss check: ", gainloss[np.isnan(gainloss)].shape)
print(" value check: ", value[np.isnan(value)].shape)
lastEmptyPriceIndex = np.zeros(adjClose.shape[0], dtype=int)
firstTrailingEmptyPriceIndex = np.zeros(adjClose.shape[0], dtype=int)

for ii in range(adjClose.shape[0]):
    # take care of special case where constant share price is inserted at beginning of series
    index = np.argmax(np.clip(np.abs(gainloss[ii, :]-1), 0, 1e-8)) - 1
    print("fist valid price and date = ", symbols[ii], " ", index, " ", datearray[index])
    lastEmptyPriceIndex[ii] = index
    activeCount[lastEmptyPriceIndex[ii]+1:] += 1

for ii in range(adjClose.shape[0]):
    # take care of special case where no quote exists at end of series
    tempQuotes = adjClose[ii, :]
    tempQuotes[np.isnan(tempQuotes)] = 1.0
    index = np.argmax(np.clip(np.abs(tempQuotes[::-1]-1), 0, 1e-8)) - 1
    if index != -1:
        firstTrailingEmptyPriceIndex[ii] = -index
        print("first trailing invalid price: index and date = ", symbols[ii], " ", firstTrailingEmptyPriceIndex[ii], " ", datearray[index])
        activeCount[firstTrailingEmptyPriceIndex[ii]:] -= 1

# --------------------------------------------------
# prepare labeled data for DL training
# - set up historical data plus actual performance one month forward
# --------------------------------------------------

first_trial = True
best_final_value = -99999
best_recent_final_value = -99999
num_periods_history = 20

try:
    for jdate in range(len(datearray)):
        year, month, day = datearray[jdate].split('-')
        datearray[jdate] = datetime.date(int(year), int(month), int(day))
except:
    pass

for itrial in range(50):

    print("\n\n\n**********************************************")
    print(" Training trial ", itrial)
    print("**********************************************\n\n\n")

    timestamp = time.strftime("_%Y-%m-%d.%H-%M")

    # update file with training parameters
    #params_filename = "pngs/"+timestamp+'.txt'
    #params_filename = "pngs/params_"+stockList+'-'+str(itrial)+"_"+timestamp+'.txt'
    params_filename = "pngs/"+str(timestamp)+"_"+stockList+"_"+str(itrial)+'.txt'
    with open(params_filename, 'a') as f:
        f.write("[training_params]\n")
        f.write("timestamp: "+timestamp+"\n")
        f.write("itrial: "+str(itrial)+"\n")
        f.write("stockList: "+stockList+"\n")

    shortest_incr = int(np.random.triangular(left=1, mode=1.3, right=5, size=1))
    #middle_incr = 2
    middle_incr = shortest_incr + int(np.random.uniform(low=1, high=6, size=1))
    #long_incr = 4
    long_incr = min(15, middle_incr + int(np.random.uniform(low=1, high=6, size=1)))
    short_period = num_periods_history*shortest_incr # days
    med_period = num_periods_history*middle_incr # months
    max_period = num_periods_history*long_incr # months
    increments = [shortest_incr, middle_incr, long_incr]
    first_history_index = adjClose.shape[1] - ((adjClose.shape[1]- max_period) / max_period)*max_period
    first_history_index = 1500
    num_stocks = 7
    print("\n ... increments (days) ", increments)
    print("\n ... analysis periods (days) ", list(np.array(increments)*num_periods_history), "\n\n")

    # update file with training parameters
    with open(params_filename, 'a') as f:
        f.write("num_stocks: "+str(num_stocks)+"\n")
        f.write("first_history_index: "+str(first_history_index)+"\n")
        f.write("first_date: "+str(datearray[first_history_index])+"\n")
        f.write("increments: "+str(increments)+"\n")
        f.write("num_periods_history: "+str(num_periods_history)+"\n")
        f.write("analysis_periods__days: "+str(list(np.array(increments)*num_periods_history))+"\n")

    # read in all stocks and randomly assign years to labeled training or validation data
    yeararray = []
    for iday in datearray[first_history_index:]:
        yeararray.append(iday.year)
    yeararray = np.array(yeararray)
    datearray_subset = np.array(datearray[first_history_index:])
    adjClose_subset = np.array(adjClose[:, first_history_index:])
    datearray_not_excluded = np.empty((1), 'float')
    adjClose_not_excluded = np.empty((adjClose_subset.shape[0]), 'float')
    datearray_excluded = np.empty((1), 'float')
    adjClose_excluded = np.empty((adjClose_subset.shape[0]), 'float')
    missing_years = np.random.uniform(low=yeararray[0], high=yeararray[-1], size=8).astype(int)
    missing_years = list(set(missing_years))
    #missing_years = [1991, 1992]
    for i, idate in enumerate(yeararray):
        if idate in missing_years:
            datearray_excluded = np.hstack((datearray_excluded, datearray_subset[i]))
            adjClose_excluded = np.vstack((adjClose_excluded, adjClose_subset[:, i]))
        else:
            datearray_not_excluded = np.hstack((datearray_not_excluded, datearray_subset[i]))
            adjClose_not_excluded = np.vstack((adjClose_not_excluded, adjClose_subset[:, i]))
    adjClose_excluded = adjClose_excluded.swapaxes(0, 1)
    adjClose_not_excluded = adjClose_not_excluded.swapaxes(0, 1)

    Xtrain, Ytrain, dates, companies = generateExamples3layer(datearray_not_excluded,
                                       adjClose_not_excluded,
                                       0,
                                       num_periods_history,
                                       increments)

    Xtest, Ytest, datesT, companiesT = generateExamples3layer(datearray_excluded,
                                       adjClose_excluded,
                                       0,
                                       num_periods_history,
                                       increments,
                                       output_incr='monthly')

    dates = np.array(dates)
    companies = np.array(companies)

    print("    ... removing nan examples in Xtrain, Ytrain ...")
    for ii in range(Xtrain.shape[0]-1, -1, -1):
        current_Xsample = Xtrain[ii, ...]
        current_Ysample = Ytrain[ii]
        current_Xsample_nans = current_Xsample[np.isnan(current_Xsample)].shape[0]
        current_Ysample_nans = current_Ysample[np.isnan(current_Ysample)].shape[0]
        if current_Xsample_nans > 0 or current_Ysample_nans > 0:
            Xtrain = np.delete(Xtrain, (ii), axis=0)
            Ytrain = np.delete(Ytrain, (ii), axis=0)
            dates = np.delete(dates, (ii), axis=0)
            companies = np.delete(companies, (ii), axis=0)

    print("    ... removing outliers in Xtrain ...")
    Xtrain_min = np.percentile(Xtrain, 0.5) * 3.
    Xtrain_max = np.percentile(Xtrain, 99.5) * 3.
    indices_to_keep = []
    for ii in range(Xtrain.shape[0]):
        if Xtrain_min < Xtrain[ii, ...].min() and Xtrain[ii, ...].max() < Xtrain_max:
            current_Xsample = Xtrain[ii, ...]
            current_Ysample = Ytrain[ii]
            if current_Xsample[np.isnan(current_Xsample)].shape[0] == 0 and \
               current_Ysample[np.isnan(current_Ysample)].shape[0] == 0:
                indices_to_keep.append(ii)
    indices_to_keep = np.array(indices_to_keep)
    Xtrain = Xtrain[indices_to_keep, ...]
    Ytrain = Ytrain[indices_to_keep]
    dates = dates[indices_to_keep]
    companies = companies[indices_to_keep]

    print("    ... removing nan examples in Xtest, Ytest ...")
    for ii in range(Xtest.shape[0]-1, -1, -1):
        current_Xsample = Xtest[ii, ...]
        current_Ysample = Ytest[ii]
        current_Xsample_nans = current_Xsample[np.isnan(current_Xsample)].shape[0]
        current_Ysample_nans = current_Ysample[np.isnan(current_Ysample)].shape[0]
        if current_Xsample_nans > 0 or current_Ysample_nans > 0:
            Xtest = np.delete(Xtest, (ii), axis=0)
            Ytest = np.delete(Ytest, (ii), axis=0)


    print("    ... removing outliers in Xtest ...")
    Xtest_min = np.percentile(Xtest, 0.5) * 3.
    Xtest_max = np.percentile(Xtest, 99.5) * 3.
    indices_to_keep = []
    for ii in range(Xtest.shape[0]):
        if Xtest_min < Xtest[ii, ...].min() and Xtest[ii, ...].max() < Xtest_max:
            current_Xsample = Xtest[ii, ...]
            current_Ysample = Ytest[ii]
            if current_Xsample[np.isnan(current_Xsample)].shape[0] == 0 and \
               current_Ysample[np.isnan(current_Ysample)].shape[0] == 0:
                indices_to_keep.append(ii)
    indices_to_keep = np.array(indices_to_keep)
    Xtest = Xtest[indices_to_keep, ...]
    Ytest = Ytest[indices_to_keep]

    print("    ... removing outliers in Ytrain ...")
    Ytrain_min = np.percentile(Ytrain, 0.5) * 3.
    Ytrain_max = np.percentile(Ytrain, 99.5) * 3.
    indices_to_keep = []
    for ii in range(Xtrain.shape[0]):
        if Ytrain_min < Ytrain[ii] < Ytrain_max:
            current_Xsample = Xtrain[ii, ...]
            current_Ysample = Ytrain[ii]
            if current_Xsample[np.isnan(current_Xsample)].shape[0] == 0 and \
               current_Ysample[np.isnan(current_Ysample)].shape[0] == 0:
                indices_to_keep.append(ii)
    indices_to_keep = np.array(indices_to_keep)
    Xtrain = Xtrain[indices_to_keep, ...]
    Ytrain = Ytrain[indices_to_keep]
    dates = dates[indices_to_keep]
    companies = companies[indices_to_keep]

    print("    ... removing outliers in Ytest ...")
    Ytest_min = np.percentile(Ytest, 0.5) * 3.
    Ytest_max = np.percentile(Ytest, 99.5) * 3.
    indices_to_keep = []
    for ii in range(Xtest.shape[0]):
        if Ytest_min < Ytest[ii] < Ytest_max:
            current_Xsample = Xtest[ii, ...]
            current_Ysample = Ytest[ii]
            if current_Xsample[np.isnan(current_Xsample)].shape[0] == 0 and \
               current_Ysample[np.isnan(current_Ysample)].shape[0] == 0:
                indices_to_keep.append(ii)
    indices_to_keep = np.array(indices_to_keep)
    Xtest = Xtest[indices_to_keep, ...]
    Ytest = Ytest[indices_to_keep]


    # drop some of the stocks randomly for each trial
    Xtrain_keep = Xtrain.copy()
    Ytrain_keep = Ytrain.copy()
    dates_keep = dates.copy()
    companies_keep = companies.copy()
    if stockList == 'Naz100':
        #number_to_drop = 10
        number_to_drop = int(np.random.uniform(low=10, high=31, size=1))
    else:
        number_to_drop = 20
    if itrial == 0:
        number_to_drop = 1
    missing_stocks = np.random.uniform(low=0, high=len(symbols), size=number_to_drop).astype(int)
    missing_stocks = np.array(list(set(missing_stocks)))
    missing_stocks.sort()
    missing_stocks = missing_stocks[::-1]

    for i in missing_stocks:
        print(" ... dropping company number ", i, symbols[i])
        Xtrain_keep = Xtrain_keep[companies_keep != i]
        Ytrain_keep = Ytrain_keep[companies_keep != i]
        dates_keep = dates_keep[companies_keep != i]
        companies_keep = companies_keep[companies_keep != i]
    print(number_to_drop, " stocks randomly dropped.  Examples before and after = ", Xtrain.shape[0], " / ", Xtrain_keep.shape[0])

    # update file with training parameters
    with open(params_filename, 'a') as f:
        f.write("missing_years: "+str(missing_years)+"\n")
        f.write("missing_stocks_indices: "+str(list(missing_stocks))+"\n")
        f.write("missing_stocks: "+str(list(np.array(symbols)[missing_stocks]))+"\n")

    # --------------------------------------------------
    # build DL model
    # --------------------------------------------------

    perform_batch_normalization = np.random.choice([True, False])
    use_dense_layers = np.random.choice([True, False])

    model = Sequential()

    #number_feature_maps = [4, 8, 16, 32, 64, 128, 128, 128]
    number_feature_maps = [4, 6, 8, 10, 12, 14, 16, 18]

    for i, nfeature in enumerate(number_feature_maps):
        if i == 0:
            model.add(Conv2D(nfeature, kernel_size=(3, 1), padding='same', strides=(1, 1), input_shape=(20, 3, 3)))
        else:
            model.add(Conv2D(nfeature, kernel_size=(3, 1), padding='same', strides=(1, 1)))
        if perform_batch_normalization is True:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        if i > 2:
            model.add(Dropout(.50))
        if _is_odd(i) or i == number_feature_maps[-1]:
            model.add(MaxPooling2D(pool_size=(2, 1)))
        print("i, model shape = ", i, model.output_shape)

    number_feature_maps2 = [16, 8, 1]
    for i, nfeature in enumerate(number_feature_maps2):
        if i == 0:
            model.add(Conv2D(nfeature, kernel_size=(1, 3), padding='same', strides=(1, 1)))
        else:
            model.add(Conv2D(nfeature, kernel_size=(1, 3), padding='same', strides=(1, 1)))
        #model.add(BatchNormalization())
        #model.add(Activation('relu'))
        if nfeature != number_feature_maps2[-1]:
            if perform_batch_normalization is True:
                model.add(BatchNormalization())
            model.add(Activation('relu'))
        if i == number_feature_maps2[-1]:
            model.add(MaxPooling2D(pool_size=(1, 2)))
        print("i, model shape = ", i, model.output_shape)
    if use_dense_layers is True:
        model.add(Dense(4))
        model.add(Dense(1))
        print("final model shape = ", model.output_shape)

    model.summary()
    #optimizer='rmsprop'
    #optimizer=adagrad(lr=0.01)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='mse')

    print("\n\n", stockList+'_patternsV6'+str(timestamp))
    weights_filename = os.path.join(os.getcwd(), 'pngs', str(timestamp)+"_"+stockList+"_"+str(itrial)+'.hdf')
    try:
        model.load_weights(weights_filename)
    except:
        pass

    # update file with training parameters
    with open(params_filename, 'a') as f:
        f.write("DL_channels: "+str(Xtrain.shape[-1])+"\n")
        f.write("perform_batch_normalization: "+str(perform_batch_normalization)+"\n")
        f.write("use_dense_layers: "+str(use_dense_layers)+"\n")


    # --------------------------------------------------
    # train DL network
    # --------------------------------------------------

    checkpoint = ModelCheckpoint(weights_filename,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')
    stop_early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    callback_list = [checkpoint, stop_early]
    num_epochs = 25

    print("\n\n ... missing_years = ", missing_years, "\n\n")
    print(" ... increments (days) ", increments)
    print(" ... analysis periods (days) ", list(np.array(increments)*num_periods_history), "\n\n")


    if first_trial or (stockList != 'SP_wo_Naz'):
        history = model.fit(Xtrain_keep, Ytrain_keep.reshape(Ytrain_keep.shape[0], 1, 1, 1),
              epochs=num_epochs,
              verbose=1,
              validation_data=(Xtest, Ytest.reshape(Ytest.shape[0], 1, 1, 1)),
              callbacks=callback_list,
              batch_size=10000)
    else:
        history = model.fit(Xtrain_keep, Ytrain_keep.reshape(Ytrain_keep.shape[0], 1, 1, 1),
              epochs=num_epochs,
              verbose=1,
              validation_data=(Xpredict, Ypredict.reshape(Ypredict.shape[0], 1, 1, 1)),
              callbacks=callback_list,
              batch_size=10000)

    # Save model when training is complete to a file
    print("Deep Learning Network trained !")


    if first_trial:
        # --------------------------------------------------
        # Import list of symbols to process for predictions
        # --------------------------------------------------

        # read list of symbols from disk.
        stockList_predict = 'Naz100'
        #stockList_predict = 'SP500'
        #stockList_predict = 'SP_wo_Naz'
        if stockList_predict == 'Naz100':
            filename_predict = os.path.join(_data_path, 'symbols', 'Naz100_Symbols.txt')                   # plotmax = 1.e10, runnum = 902
        elif stockList_predict == 'SP500' or stockList_predict == 'SP_wo_Naz':
            filename_predict = os.path.join(_data_path, 'symbols', 'SP500_Symbols.txt')                   # plotmax = 1.e10, runnum = 902

        # --------------------------------------------------
        # Get quotes for each symbol in list
        # process dates.
        # Clean up quotes.
        # Make a plot showing all symbols in list
        # --------------------------------------------------

        ## update quotes from list of symbols
        (symbols_directory, symbols_file) = os.path.split(filename_predict)
        basename, extension = os.path.splitext(symbols_file)
        print((" symbols_directory = ", symbols_directory))
        print(" symbols_file = ", symbols_file)
        print("symbols_directory, symbols.file = ", symbols_directory, symbols_file)
        ###############################################################################################
        do_update = False
        if do_update is True:
            UpdateHDF5(symbols_directory, symbols_file)  ### assume hdf is already up to date
        adjClose_predict, symbols_predict, datearray_predict, _, _ = loadQuotes_fromHDF(filename_predict)

        Xpredict, Ypredict, dates_predict, companies_predict = generateExamples3layer(datearray_predict,
                                           adjClose_predict,
                                           first_history_index,
                                           num_periods_history,
                                           increments,
                                           output_incr='monthly')

    # update file with training parameters
    with open(params_filename, 'a') as f:
        f.write("stockList_predict: "+stockList_predict+"\n")
        f.write("weights_filename: "+weights_filename+"\n")

    # --------------------------------------------------
    # make predictions monthly for backtesting
    # - there might be some bias since entire preiod
    #   has data used for training
    # --------------------------------------------------

    #weights_filename = os.path.join(os.getcwd(), 'patternsV5.hdf')
    ##weights_filename = os.path.join(os.getcwd(), 'patternsV5_2017-11-18.23-28.hdf')
    try:
        model.load_weights(weights_filename)
    except:
        pass

    dates_predict = np.array(dates_predict)
    companies_predict = np.array(companies_predict)

    # crossplot predictions against actual preformance
    plt.figure(5, figsize=(14, 10))
    plt.clf()
    plt.grid(True)
    _plt5_xmin = []
    _plt5_xmax = []
    _plt5_ymin = []
    _plt5_ymax = []

    #for num_stocks in range(5, 9):
    for inum_stocks in [num_stocks]:
        cumu_system = [10000.0]
        cumu_system_worst = [10000.0]
        cumu_BH = [10000.0]
        plotdates = [dates_predict[0]]
        for i, idate in enumerate(dates_predict[1:]):
            if idate != dates[-1] and companies_predict[i] < companies_predict[i-1]:
                # show predictions for (single) last date
                _Xtrain = Xpredict[dates_predict == idate]
                _dates = np.array(dates_predict[dates_predict == idate])
                _companies = np.array(companies_predict[dates_predict == idate])
                _forecast = model.predict(_Xtrain)[:, 0, 0, 0]
                _symbols = np.array(symbols_predict)

                indices = _forecast.argsort()
                sorted_forecast = _forecast[indices]
                sorted_symbols = _symbols[indices]

                try:
                    _Ytrain = Ypredict[dates_predict == idate]
                    sorted_Ytrain = _Ytrain[indices]
                    BH_gain = sorted_Ytrain.mean()
                    plt.plot(sorted_Ytrain[:-num_stocks], sorted_forecast[:-num_stocks], 'k.', markersize=3)
                    plt.plot(sorted_Ytrain[-num_stocks:], sorted_forecast[-num_stocks:], 'r.', markersize=3)
                    _plt5_xmin.append(sorted_Ytrain.min())
                    _plt5_xmax.append(sorted_Ytrain.max())
                    _plt5_ymin.append(sorted_forecast.min())
                    _plt5_ymax.append(sorted_forecast.max())
                except:
                    BH_gain = 1.0

                avg_gain = sorted_Ytrain[-inum_stocks:].mean()
                avg_gain_worst = sorted_Ytrain[:inum_stocks].mean()

                print(" ... date, system_gain, B&H_gain = ",
                      idate,
                      format(avg_gain, '3.1%'), format(BH_gain, '3.1%'),
                      sorted_symbols[-inum_stocks:])
                cumu_system.append(cumu_system[-1] * (1.+avg_gain))
                cumu_system_worst.append(cumu_system_worst[-1] * (1.+avg_gain_worst))
                cumu_BH.append(cumu_BH[-1] * (1.+BH_gain))
                plotdates.append(idate)
        print(" ...system, B&H = ", format(cumu_system[-1], '10,.0f'), format(cumu_BH[-1], '10,.0f'))

        # plot results
        num_months = int(12*4.6)
        #system_label = str(itrial)+" "+timestamp+" "+str(missing_years)
        system_label = str(itrial)+" "+timestamp+" "+str(missing_years)+" "+str(increments)
        #system_label = str(inum_stocks)+" "+timestamp+" "+str(missing_years)
        plt.figure(3, figsize=(14, 10))
        #if itrial==0:
        #if inum_stocks==1:
        if first_trial or itrial%1 == 0:
            plt.clf()
            plt.plot(plotdates, cumu_BH, 'r-', lw=3, label='B&H')
        #plt.plot(plotdates, cumu_system, 'k-', label='%s %s' % (str(timestamp), str(missing_years)))
        plt.plot(plotdates, cumu_system, 'k-', label=system_label)
        plt.plot(plotdates, cumu_system_worst, 'b-', lw=.25)
        if not first_trial:
            plt.plot(plotdates, best_series, '-', color='peru', lw=2., alpha=.8, label='best '+best_model)
            plt.plot(plotdates, best_sharpe_series, 'o-', color='c', lw=1., markersize=3.5, label='best_sharpe '+best_sharpe_model)
            plt.text(plotdates[-1]+datetime.timedelta(200), best_series[-1], str(best_trial), color='peru')
            plt.text(plotdates[-1]+datetime.timedelta(266), best_sharpe_series[-1], str(best_sharpe_trial), color='c')
        plt.grid(True)
        plt.yscale('log')
        plt.legend()
        plt.title('Train on '+stockList+' w/o '+str(len(missing_stocks))+' random stocks\nPredict on all '+stockList_predict+' stocks\n'+str(missing_stocks))
        #plt.text(plotdates[-1], cumu_system[-1], str(itrial))
        plt.text(plotdates[-1], cumu_system[-1], str(itrial))
        plt.text(plotdates[-1]+datetime.timedelta(100), cumu_system_worst[-1], str(itrial), color='b')
        plt.text(plotdates[0]+datetime.timedelta(100), 7000., "perform_batch_normalization: "+str(perform_batch_normalization))
        plt.text(plotdates[0]+datetime.timedelta(2500), 7000., "use_dense_layers: "+str(use_dense_layers))
        plt.savefig("pngs/"+timestamp+"_"+stockList+'_'+str(itrial)+"_fig-3"+'.png', format='png')

        # plot results
        plt.figure(4, figsize=(14, 10))
        #if itrial==0:
        #if inum_stocks==1:
        if first_trial or itrial%1 == 0:
            plt.clf()
            plt.plot(plotdates[-num_months:], cumu_BH[-num_months:]/cumu_BH[-num_months]*10000., 'r-', lw=3, label='B&H')
        plt.plot(plotdates[-num_months:], cumu_system[-num_months:]/cumu_system[-num_months]*10000., 'k-', label=system_label)
        plt.plot(plotdates[-num_months:], cumu_system_worst[-num_months:]/cumu_system_worst[-num_months]*10000., 'b-', lw=.25)
        if not first_trial:
            plt.plot(plotdates[-num_months:], best_recent_series, '-', color='peru', lw=2., alpha=.8, label='best_recent '+best_recent_model)
            plt.plot(plotdates[-num_months:], best_sharpe_recent_series, 'o-', color='c', lw=1., markersize=3.5, label='best_sharpe_recent '+best_sharpe_recent_model)
            plt.text(plotdates[-1]+datetime.timedelta(40), best_recent_series[-1], str(best_recent_trial), color='peru')
            plt.text(plotdates[-1]+datetime.timedelta(50), best_sharpe_recent_series[-1], str(best_sharpe_recent_trial), color='c')
        plt.grid(True)
        plt.yscale('log')
        plt.legend(loc='upper left')
        #plt.title('Train on SP500 w/o 20 random stocks\nPredict on all SP500 stocks\n'+str(missing_stocks))
        plt.title('Train on '+stockList+' w/o '+str(len(missing_stocks))+' random stocks\nPredict on all '+stockList_predict+' stocks\n'+str(missing_stocks))
        #plt.text(plotdates[-1], cumu_system[-1]/cumu_system[-num_months]*10000., str(itrial))
        plt.text(plotdates[-1], cumu_system[-1]/cumu_system[-num_months]*10000., str(itrial))
        plt.text(plotdates[-1]+datetime.timedelta(20), cumu_system_worst[-1]/cumu_system_worst[-num_months]*10000., str(itrial), color='b')
        plt.text(plotdates[-num_months]+datetime.timedelta(100), 9500., "perform_batch_normalization: "+str(perform_batch_normalization))
        plt.text(plotdates[-num_months]+datetime.timedelta(800), 9500., "use_dense_layers: "+str(use_dense_layers))
        #plt.savefig("pngs/fig-4_"+stockList+'_'+str(itrial)+"_"+timestamp+'.png', format='png')
        plt.savefig("pngs/"+timestamp+"_"+stockList+'_'+str(itrial)+"_fig-4"+'.png', format='png')

        # crossplot predictions against actual preformance
        plt.figure(5, figsize=(14, 10))
        plt.title('Crossplot of actual and forecast Gain/Loss one month forward')
        _plt5_xmin = np.percentile(_plt5_xmin, 20)
        _plt5_xmax = np.percentile(_plt5_xmax, 80)
        _plt5_ymin = np.percentile(_plt5_ymin, 5)
        _plt5_ymax = np.percentile(_plt5_ymax, 95)
        #_plt5_xmin = -.5
        #_plt5_xmax = .5
        #_plt5_ymin = -.5
        #_plt5_ymax = .5
        plt.xlim((_plt5_xmin, _plt5_xmax))
        plt.ylim((_plt5_ymin, _plt5_ymax))
        plt.xlabel('actual')
        plt.ylabel('forecast')
        #plt.savefig("pngs/"+stockList+'_'+str(inum_stocks)+'_Crossplot_'+str(itrial)+"_"+timestamp+"_"+str(missing_years)+str(increments)+'.png', format='png')
        plt.savefig("pngs/"+timestamp+"_"+stockList+'_'+str(itrial)+"_crossplot"+'.png', format='png')

        forecast_date = datearray[-1]
        forecast_date = datetime.date(2017, 11, 1)
        _Xtrain_today, _dates_today, _companies_today = \
                         generatePredictionInput3layer(forecast_date,
                                                       datearray_predict,
                                                       adjClose_predict,
                                                       first_history_index,
                                                       num_periods_history,
                                                       increments)

        _companies_today = np.array(_companies_today)
        _forecast_today = model.predict(_Xtrain_today)[:, 0, 0, 0]
        indices = _forecast_today.argsort()
        sorted_companynumbers_today = _companies_today[indices]
        sorted_symbols_today = np.array(symbols)[sorted_companynumbers_today]
        print("\n\n ... Predictions for ",
              _dates_today[-1],
              ":\n ... Companies to buy = ",
              sorted_symbols_today[-inum_stocks:],
              "\n\n\n")

        # compute performance stats
        sharpe_BH = allstats(np.array(cumu_BH)).monthly_sharpe()
        sharpe_system = allstats(np.array(cumu_system)).monthly_sharpe()
        sharpe_system_worst = allstats(np.array(cumu_system_worst)).monthly_sharpe()

        sharpe_recent_BH = allstats(cumu_BH[-num_months:]/cumu_BH[-num_months]*10000.).monthly_sharpe()
        sharpe_recent_system = allstats(cumu_system[-num_months:]/cumu_system[-num_months]*10000.).monthly_sharpe()
        sharpe_recent_system_worst = allstats(cumu_system_worst[-num_months:]/cumu_system_worst[-num_months]*10000.).monthly_sharpe()

        # keep track of best performance
        if first_trial:
            best_model = weights_filename
            best_final_value = cumu_system[-1]
            best_recent_model = weights_filename
            best_recent_final_value = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.)[-1]
            best_trial = itrial
            best_recent_trial = itrial
            best_sharpe_model = weights_filename
            best_sharpe_ratio = sharpe_system
            best_sharpe_final_value = cumu_system[-1]
            best_sharpe_recent_model = weights_filename
            best_sharpe_recent_ratio = sharpe_recent_system
            best_sharpe_recent_final_value = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.)[-1]
            best_sharpe_trial = itrial
            best_sharpe_recent_trial = itrial
            best_series = cumu_system.copy()
            best_recent_series = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.).copy()
            best_sharpe_series = cumu_system.copy()
            best_sharpe_recent_series = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.).copy()
        if not first_trial and cumu_system[-1] > best_final_value:
            best_model = weights_filename
            best_final_value = cumu_system[-1]
            best_trial = itrial
            best_series = cumu_system.copy()
        if not first_trial and cumu_system[-1]/cumu_system[-num_months]*10000. > best_recent_final_value:
            best_recent_model = weights_filename
            best_recent_final_value = cumu_system[-1]/cumu_system[-num_months]*10000.
            best_recent_trial = itrial
            best_recent_series = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.).copy()
        if not first_trial and sharpe_system > best_sharpe_ratio:
            best_sharpe_model = weights_filename
            best_sharpe_ratio = sharpe_system
            best_sharpe_final_value = cumu_system[-1]
            best_sharpe_trial = itrial
            best_sharpe_series = cumu_system.copy()
        if not first_trial and sharpe_recent_system > best_sharpe_recent_ratio:
            best_sharpe_recent_model = weights_filename
            best_sharpe_recent_ratio = sharpe_recent_system
            best_sharpe_recent_final_value = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.)[-1]
            best_sharpe_recent_trial = itrial
            best_sharpe_recent_series = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.).copy()

        print("\n\n ... Best model: trial = ", best_trial, " file = ", best_model)
        print(" ... Best final value = ", format(best_final_value, '10,.0f'))
        print("\n ... Best recent model: trial = ", best_recent_trial, " file = ", best_recent_model)
        print(" ... Best recent final value = ", format(best_recent_final_value, '10,.0f'), "\n\n\n")

        print("\n\n ... Best sharpe model: trial = ", best_sharpe_trial, " file = ", best_sharpe_model)
        print(" ... Best sharpe ratio = ", format(best_sharpe_ratio, '6.3f'))
        print(" ... Best sharpe value = ", format(best_sharpe_final_value, '10,.0f'))
        print("\n ... Best sharpe recent model: trial = ", best_sharpe_recent_trial, " file = ", best_sharpe_recent_model)
        print(" ... Best sharpe recent_ratio = ", format(best_sharpe_recent_ratio, '6.3f'))
        print(" ... Best sharpe_recent value = ", format(best_sharpe_recent_final_value, '10,.0f'), "\n\n\n")

        print("\ntraining history = ", history.history['val_loss'][-1], history.history['loss'][-1], "\n\n\n")

        # update file with training parameters
        _validation_loss = np.min(history.history['val_loss'])
        saved_model_index = np.argmin(history.history['val_loss'])
        _loss = list(history.history['loss'])[saved_model_index]
        with open(params_filename, 'a') as f:
            f.write("_validation_loss: "+str(_validation_loss)+"\n")
            f.write("_loss: "+str(_loss)+"\n")
            f.write("_final_value_BH: "+format(cumu_BH[-1], '10,.0f')+"\n")
            f.write("_final_value_system: "+format(cumu_system[-1], '10,.0f')+"\n")
            f.write("_final_recent_value_BH: "+format(cumu_BH[-1]/cumu_BH[-num_months]*10000., '10,.0f')+"\n")
            f.write("_final_recent_value_system: "+format(cumu_system[-1]/cumu_system[-num_months]*10000., '10,.0f')+"\n")
            f.write("_sharpe_ratio_BH: "+str(sharpe_BH)+"\n")
            f.write("_sharpe_ratio_system: "+str(sharpe_system)+"\n")
            f.write("_sharpe_ratio_system_worst: "+str(sharpe_system_worst)+"\n")
            f.write("_sharpe_ratio_recent_BH: "+str(sharpe_recent_BH)+"\n")
            f.write("_sharpe_ratio_recent_system: "+str(sharpe_recent_system)+"\n")
            f.write("_sharpe_ratio_recent_system_worst: "+str(sharpe_recent_system_worst)+"\n")
            f.write("number_feature_maps: "+str(number_feature_maps)+"\n")

        first_trial = False

        sleep(15)



