###
### A Multilayer Perceptron implementation example using TensorFlow library.
###

import os
import time
from time import sleep
import datetime
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GaussianNoise
from keras.initializers import Initializer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam, Adagrad, Nadam

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

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
                                  generateExamples3layerGen, \
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
stockList = 'SP_wo_Naz'
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
    print("first valid price and date = ", symbols[ii], " ", index, " ", datearray[index])
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

for itrial in range(4,25):

    start_time = time.time()

    print("\n\n\n**********************************************")
    print(" Training trial ", itrial)
    print("**********************************************\n\n\n")

    # --------------------------------------------------
    # set up randomly selected parameters
    # --------------------------------------------------
    timestamp = time.strftime("_%Y-%m-%d.%H-%M")
    shortest_incr = int(np.random.triangular(left=1, mode=3, right=5.8, size=1))
    long_incr = min(15, shortest_incr + int(np.random.triangular(left=2, mode=7.5, right=13, size=1)))
    incr_incr = max(1, int(float(long_incr - shortest_incr) / 3.))
    #first_history_index = adjClose.shape[1] - ((adjClose.shape[1]- max_period) / max_period)*max_period
    first_history_index = 1500
    num_stocks = 7

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

    perform_batch_normalization = np.random.choice([True, False, False, False])
    #use_dense_layers = np.random.choice([True, False])
    dense_factor = np.random.triangular(left=2., mode=3.5, right=8.)
    use_leaky_relu = np.random.choice([True, False])
    use_separable = np.random.choice([True, True, True, True, False])
    leaky_relu_alpha = np.random.triangular(left=0.1, mode=.375, right=.7)
    #feature_map_factor = np.random.triangular(left=0., mode=.3, right=3)

    #number_feature_maps = [4, 8, 16, 32, 64, 128, 128, 128]
    #number_feature_maps = [4, 6, 8, 10, 12, 14, 16, 18]
    feature_map_factor = np.random.choice([1,2,3,4,5,6,7,8]) * 5
    #number_feature_maps = np.array([4, 6, 8, 10, 12, 14, 16, 18]) * feature_map_factor
    number_feature_maps = np.ones(int(np.random.triangular(3,3,15.99)), 'int') * feature_map_factor

    optimizer_choice = np.random.choice(['RMSprop',
                                         'RMSprop',
                                         'RMSprop',
                                         'Adam',
                                         'Adagrad',
                                         'Nadam'])

    loss_function = np.random.choice(['mse',
                                  'mean_absolute_error',
                                  'mean_absolute_percentage_error',
                                  #'kullback_leibler_divergence',
                                  'mean_squared_logarithmic_error'])
    callback_mode = 'auto'
    '''
    if loss_function=='kullback_leibler_divergence':
        callback_mode = 'max'
    else:
        callback_mode = 'min'
    '''

    learning_rate = 10. ** np.random.triangular(-4.1, -3.15, -2.25)
    dropout_pct = np.random.triangular(0.05, .35, 0.65)

    # --------------------------------------------------
    # compute additonal training parameters
    # --------------------------------------------------

    # update file with training parameters
    #params_filename = "pngs/"+timestamp+'.txt'
    #params_filename = "pngs/params_"+stockList+'-'+str(itrial)+"_"+timestamp+'.txt'
    params_filename = "pngs/"+str(timestamp)+"_"+stockList+"_"+str(itrial)+'.txt'
    with open(params_filename, 'a') as f:
        f.write("[training_params]\n")
        try:
            f.write("python_file: "+__file__+"\n")
        except:
            pass
        f.write("timestamp: "+timestamp+"\n")
        f.write("itrial: "+str(itrial)+"\n")
        f.write("stockList: "+stockList+"\n")

    '''
    shortest_incr = int(np.random.triangular(left=1, mode=1.3, right=5, size=1))
    #middle_incr = 2
    middle_incr = shortest_incr + int(np.random.uniform(low=1, high=6, size=1))
    #long_incr = 4
    long_incr = min(15, middle_incr + int(np.random.uniform(low=1, high=6, size=1)))
    short_period = num_periods_history*shortest_incr # days
    med_period = num_periods_history*middle_incr # months
    max_period = num_periods_history*long_incr # months
    increments = [shortest_incr, middle_incr, long_incr]
    '''
    increments = np.arange(shortest_incr, long_incr+1.e-5, incr_incr).astype('int')
    print(" ... increments = ", increments)

    max_period = num_periods_history*increments[-1] # months
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

    # --------------------------------------------------
    # read in all stocks and randomly assign years to labeled training or validation data
    # --------------------------------------------------

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
    #if stockList == 'SP_wo_Naz':
    #    missing_years = [2006, 2013]
    for i, idate in enumerate(yeararray):
        if idate in missing_years:
            datearray_excluded = np.hstack((datearray_excluded, datearray_subset[i]))
            adjClose_excluded = np.vstack((adjClose_excluded, adjClose_subset[:, i]))
        else:
            datearray_not_excluded = np.hstack((datearray_not_excluded, datearray_subset[i]))
            adjClose_not_excluded = np.vstack((adjClose_not_excluded, adjClose_subset[:, i]))
    adjClose_excluded = adjClose_excluded.swapaxes(0, 1)
    adjClose_not_excluded = adjClose_not_excluded.swapaxes(0, 1)

    Xtrain, Ytrain, dates, companies = generateExamples3layerGen(datearray_not_excluded,
                                       adjClose_not_excluded,
                                       0,
                                       num_periods_history,
                                       increments)

    Xtest, Ytest, datesT, companiesT = generateExamples3layerGen(datearray_excluded,
                                       adjClose_excluded,
                                       0,
                                       num_periods_history,
                                       increments,
                                       output_incr='monthly')

    dates = np.array(dates)
    companies = np.array(companies)
    #break

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

    for i in missing_stocks:
        print(" ... dropping company number ", i, symbols[i])
        Xtrain_keep = Xtrain_keep[companies_keep != i]
        Ytrain_keep = Ytrain_keep[companies_keep != i]
        dates_keep = dates_keep[companies_keep != i]
        companies_keep = companies_keep[companies_keep != i]
    print(number_to_drop, " stocks randomly dropped.  Examples before and after = ", Xtrain.shape[0], " / ", Xtrain_keep.shape[0])

    # shuffle training data
    shuffle_indices = list(np.arange(Xtrain_keep.shape[0]))
    random_ordering = list(np.random.uniform(low=0, high=Xtrain_keep.shape[0], size=Xtrain_keep.shape[0]))
    shuffling_indices = [x for _, x in sorted(zip(random_ordering,shuffle_indices))]
    Xtrain_keep = Xtrain_keep[shuffling_indices, ...]
    Ytrain_keep = Ytrain_keep[shuffling_indices]
    dates_keep = dates_keep[shuffling_indices]
    companies_keep = companies_keep[shuffling_indices]
    del shuffle_indices
    del random_ordering
    del shuffling_indices
    print(" Xtrain_keep and Ytrain_keep randomly shuffled.")

    # update file with training parameters
    with open(params_filename, 'a') as f:
        f.write("missing_years: "+str(missing_years)+"\n")
        f.write("missing_stocks_indices: "+str(list(missing_stocks))+"\n")
        f.write("missing_stocks: "+str(list(np.array(symbols)[missing_stocks]))+"\n")

    # --------------------------------------------------
    # build DL model
    # --------------------------------------------------

    model = Sequential()

    for i, nfeature in enumerate(number_feature_maps):
        if i == 0:
            model.add(Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             kernel_initializer='lecun_uniform', bias_initializer='zeros',
                             input_shape=(Xtrain.shape[1], Xtrain.shape[2], 3)))
        else:
            if perform_batch_normalization is True:
                model.add(Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             use_bias=False))
                model.add(BatchNormalization())
            else:
                model.add(Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             use_bias=True))
        #model.add(GaussianNoise(.0001))
        if use_leaky_relu is False:
            model.add(Activation('relu'))
        else:
            model.add(LeakyReLU(alpha=leaky_relu_alpha))
        if i > 2:
            model.add(Dropout(dropout_pct))
        #if _is_odd(i) or i == number_feature_maps[-1]:
        #    model.add(MaxPooling2D(pool_size=(2, 1)))
        print("i, model shape = ", i, model.output_shape)

    nfeature = 16
    for i in range(2):
        if use_separable == False:
            model.add(Conv2D(nfeature, kernel_size=(1, 3), padding='same', strides=(1, 1),
                             data_format='channels_last', use_bias=False))
        else:
            model.add(Dense(16))
            print("i, before separableConv2D, model shape = ", i, model.output_shape)
            model.add(SeparableConv2D(nfeature, kernel_size=(1, 3), padding='same',
                                      strides=(1, 1), data_format='channels_last',
                                      depth_multiplier=1, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(int(64./dense_factor), kernel_initializer='lecun_uniform', bias_initializer='zeros'))
    model.add(Dense(int(32./dense_factor)))
    model.add(Dense(int(16./dense_factor)))
    model.add(Dense(1))
    print("final model shape = ", model.output_shape)

    model.summary()
    print("feature_map_factor = ", feature_map_factor)

    if optimizer_choice == 'RMSprop':
        optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    elif optimizer_choice == 'Adam':
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif optimizer_choice == 'Adagrad':
        optimizer = Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0)
    elif optimizer_choice == 'Nadam':
        optimizer = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    model.compile(optimizer=optimizer, loss=loss_function)
    print("loss function = ",loss_function)

    print("\n\n", stockList+'_patternsV6'+str(timestamp))
    weights_filename = os.path.join(os.getcwd(), 'pngs', str(timestamp)+"_"+stockList+"_"+str(itrial)+'.hdf')
    model_json_filename = os.path.join(os.getcwd(), 'pngs', str(timestamp)+"_"+stockList+"_"+str(itrial)+'.json')

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_json_filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_filename)
    print("Saved model to disk")

    try:
        model.load_weights(weights_filename)
    except:
        pass

    # update file with training parameters
    with open(params_filename, 'a') as f:
        f.write("DL_channels: "+str(Xtrain.shape[-1])+"\n")
        f.write("number_feature_maps = "+str(number_feature_maps)+"\n")
        f.write("feature_map_factor = "+str(feature_map_factor)+"\n")
        f.write("perform_batch_normalization: "+str(perform_batch_normalization)+"\n")
        f.write("dense_factor: "+str(dense_factor)+"\n")
        f.write("use_leaky_relu: "+str(use_leaky_relu)+"\n")
        f.write("use_separable: "+str(use_separable)+"\n")
        f.write("leaky_relu_alpha: "+str(leaky_relu_alpha)+"\n")
        f.write("optimizer_choice: "+optimizer_choice+"\n")
        f.write("learning_rate: "+str(learning_rate)+"\n")
        f.write("loss function: "+loss_function+"\n")

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

    Xpredict, Ypredict, dates_predict, companies_predict = generateExamples3layerGen(datearray_predict,
                                       adjClose_predict,
                                       first_history_index,
                                       num_periods_history,
                                       increments,
                                       output_incr='monthly')

    # update file with training parameters
    with open(params_filename, 'a') as f:
        f.write("stockList_predict: "+stockList_predict+"\n")
        f.write("weights_filename: "+weights_filename+"\n")
        f.write("model_json_filename: "+model_json_filename+"\n")

    # --------------------------------------------------
    # train DL network
    # --------------------------------------------------

    checkpoint = ModelCheckpoint(weights_filename,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode=callback_mode)
    stop_early = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode=callback_mode)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    callback_list = [checkpoint, stop_early, reduce_lr]
    num_epochs = 50

    print("\n\n ... missing_years = ", missing_years, "\n\n")
    print(" ... increments (days) ", increments)
    print(" ... analysis periods (days) ", list(np.array(increments)*num_periods_history), "\n\n")


    if first_trial or (stockList != 'SP_wo_Naz'):
        history = model.fit(Xtrain_keep, Ytrain_keep.reshape(Ytrain_keep.shape[0], 1),
              epochs=num_epochs,
              verbose=1,
              validation_data=(Xtest, Ytest.reshape(Ytest.shape[0], 1)),
              callbacks=callback_list,
              batch_size=500)
    else:
        history = model.fit(Xtrain_keep, Ytrain_keep.reshape(Ytrain_keep.shape[0], 1),
              epochs=num_epochs,
              verbose=1,
              validation_data=(Xpredict, Ypredict.reshape(Ypredict.shape[0], 1)),
              callbacks=callback_list,
              batch_size=500)

    # Save model when training is complete to a file
    print("Deep Learning Network trained !")

    # update file with training parameters
    with open(params_filename, 'a') as f:
        f.write("loss: "+str(history.history['loss'][-1])+"\n")
        f.write("val_loss: "+str(history.history['val_loss'][-1])+"\n")
        f.write("training_epochs: "+str(len(history.history['val_loss']))+"\n")

    # --------------------------------------------------
    # make predictions monthly for backtesting
    # - there might be some bias since entire preiod
    #   has data used for training
    # --------------------------------------------------

    try:
        # load json and create model
        json_file = open(model_json_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    except:
        pass
    try:
        model.load_weights(weights_filename)
    except:
        pass

    dates_predict = np.array(dates_predict)
    companies_predict = np.array(companies_predict)

    # crossplot predictions against actual preformance
    plt.close(5)
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
        _forecast_mean = []
        _forecast_median = []
        _forecast_stdev = []
        for i, idate in enumerate(dates_predict[1:]):
            if idate != dates[-1] and companies_predict[i] < companies_predict[i-1]:
                # show predictions for (single) last date
                _Xtrain = Xpredict[dates_predict == idate]
                _dates = np.array(dates_predict[dates_predict == idate])
                _companies = np.array(companies_predict[dates_predict == idate])
                _forecast = model.predict(_Xtrain)[:, 0]
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

                _forecast_mean.append(_forecast.mean())
                _forecast_median.append(np.median(_forecast))
                _forecast_stdev.append(_forecast.std())

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
        _forecast_mean -= np.mean(_forecast_mean)
        _forecast_median -= np.median(_forecast_median)
        num_months_medium = int(12*13.75)
        num_months = int(12*4.75)
        #system_label = str(itrial)+" "+timestamp+" "+str(missing_years)
        system_label = str(itrial)+" "+timestamp+" "+str(missing_years)+" "+str(increments)
        #system_label = str(inum_stocks)+" "+timestamp+" "+str(missing_years)
        plt.close(2)
        plt.figure(2, figsize=(14, 10))
        if first_trial or itrial%1 == 0:
            plt.clf()
            subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,3])
            plt.subplot(subplotsize[0])
        if first_trial or itrial%1 == 0:
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
        plt.text(plotdates[0]+datetime.timedelta(2500), 7000., "dense_factors: "+str(dense_factor))
        plt.text(plotdates[0]+datetime.timedelta(5000), 7000., "loss function: "+loss_function)
        plt.subplot(subplotsize[1])
        plt.grid(True)
        plt.plot(plotdates[1:], _forecast_mean, label='forecast_mean')
        plt.plot(plotdates[1:], _forecast_median, label='forecast_median')
        plt.plot(plotdates[1:], _forecast_stdev, label='forecast_stdev')
        plt.legend()
        plt.savefig("pngs/"+timestamp+"_"+stockList+'_'+str(itrial)+"_fig-2"+'.png', format='png')

        # plot medium-term results
        plt.close(3)
        plt.figure(3, figsize=(14, 10))
        if first_trial or itrial%1 == 0:
            plt.clf()
            subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,3])
            plt.subplot(subplotsize[0])
        if first_trial or itrial%1 == 0:
            plt.plot(plotdates[-num_months_medium:], cumu_BH[-num_months_medium:]/cumu_BH[-num_months_medium]*10000., 'r-', lw=3, label='B&H')
        plt.plot(plotdates[-num_months_medium:], cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000., 'k-', label=system_label)
        plt.plot(plotdates[-num_months_medium:], cumu_system_worst[-num_months_medium:]/cumu_system_worst[-num_months_medium]*10000., 'b-', lw=.25)
        if not first_trial:
            plt.plot(plotdates[-num_months_medium:], best_medium_series, '-', color='peru', lw=2., alpha=.8, label='best_recent '+best_recent_model)
            plt.plot(plotdates[-num_months_medium:], best_sharpe_medium_series, 'o-', color='c', lw=1., markersize=3.5, label='best_sharpe_recent '+best_sharpe_recent_model)
            plt.text(plotdates[-1]+datetime.timedelta(40), best_medium_series[-1], str(best_medium_trial), color='peru')
            plt.text(plotdates[-1]+datetime.timedelta(50), best_sharpe_medium_series[-1], str(best_sharpe_medium_trial), color='c')
        plt.grid(True)
        plt.yscale('log')
        plt.legend(loc='upper left')
        #plt.title('Train on SP500 w/o 20 random stocks\nPredict on all SP500 stocks\n'+str(missing_stocks))
        plt.title('Train on '+stockList+' w/o '+str(len(missing_stocks))+' random stocks\nPredict on all '+stockList_predict+' stocks\n'+str(missing_stocks))
        #plt.text(plotdates[-1], cumu_system[-1]/cumu_system[-num_months_medium]*10000., str(itrial))
        plt.text(plotdates[-1], cumu_system[-1]/cumu_system[-num_months_medium]*10000., str(itrial))
        plt.text(plotdates[-1]+datetime.timedelta(20), cumu_system_worst[-1]/cumu_system_worst[-num_months_medium]*10000., str(itrial), color='b')
        plt.text(plotdates[-num_months_medium]+datetime.timedelta(100), 9500., "perform_batch_normalization: "+str(perform_batch_normalization))
        plt.text(plotdates[-num_months_medium]+datetime.timedelta(700), 9500., "dense_factor: "+str(dense_factor))
        plt.text(plotdates[-num_months_medium]+datetime.timedelta(1200), 9500., "loss function: "+loss_function)
        #plt.savefig("pngs/fig-4_"+stockList+'_'+str(itrial)+"_"+timestamp+'.png', format='png')
        plt.subplot(subplotsize[1])
        plt.grid(True)
        plt.plot(plotdates[-num_months_medium:], _forecast_mean[-num_months_medium:], label='forecast_mean')
        plt.plot(plotdates[-num_months_medium:], _forecast_median[-num_months_medium:], label='forecast_median')
        plt.plot(plotdates[-num_months_medium:], _forecast_stdev[-num_months_medium:], label='forecast_stdev')
        plt.legend(loc='upper left')
        plt.savefig("pngs/"+timestamp+"_"+stockList+'_'+str(itrial)+"_fig-3"+'.png', format='png')

        # plot short-term results
        plt.close(4)
        plt.figure(4, figsize=(14, 10))
        if first_trial or itrial%1 == 0:
            plt.clf()
            subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,3])
            plt.subplot(subplotsize[0])
        if first_trial or itrial%1 == 0:
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
        plt.text(plotdates[-num_months]+datetime.timedelta(700), 9500., "dense_factor: "+str(dense_factor))
        plt.text(plotdates[-num_months]+datetime.timedelta(1200), 9500., "loss function: "+loss_function)
        #plt.savefig("pngs/fig-4_"+stockList+'_'+str(itrial)+"_"+timestamp+'.png', format='png')
        plt.subplot(subplotsize[1])
        plt.grid(True)
        plt.plot(plotdates[-num_months:], _forecast_mean[-num_months:], label='forecast_mean')
        plt.plot(plotdates[-num_months:], _forecast_median[-num_months:], label='forecast_median')
        plt.plot(plotdates[-num_months:], _forecast_stdev[-num_months:], label='forecast_stdev')
        plt.legend(loc='upper left')
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
        _forecast_today = model.predict(_Xtrain_today)[:, 0]
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

        sharpe_medium_BH = allstats(cumu_BH[-num_months_medium:]/cumu_BH[-num_months_medium]*10000.).monthly_sharpe()
        sharpe_medium_system = allstats(cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000.).monthly_sharpe()
        sharpe_medium_system_worst = allstats(cumu_system_worst[-num_months_medium:]/cumu_system_worst[-num_months_medium]*10000.).monthly_sharpe()

        sharpe_recent_BH = allstats(cumu_BH[-num_months:]/cumu_BH[-num_months]*10000.).monthly_sharpe()
        sharpe_recent_system = allstats(cumu_system[-num_months:]/cumu_system[-num_months]*10000.).monthly_sharpe()
        sharpe_recent_system_worst = allstats(cumu_system_worst[-num_months:]/cumu_system_worst[-num_months]*10000.).monthly_sharpe()

        # keep track of best performance
        if first_trial:

            best_trial = itrial
            best_model = weights_filename
            best_final_value = cumu_system[-1]
            best_series = cumu_system.copy()

            best_medium_trial = itrial
            best_medium_model = weights_filename
            best_medium_final_value = (cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000.)[-1]
            best_medium_series = (cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000.).copy()

            best_recent_trial = itrial
            best_recent_model = weights_filename
            best_recent_final_value = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.)[-1]
            best_recent_series = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.).copy()

            best_sharpe_trial = itrial
            best_sharpe_model = weights_filename
            best_sharpe_ratio = sharpe_system
            best_sharpe_final_value = cumu_system[-1]
            best_sharpe_series = cumu_system.copy()

            best_sharpe_medium_trial = itrial
            best_sharpe_medium_model = weights_filename
            best_sharpe_medium_ratio = sharpe_medium_system
            best_sharpe_medium_final_value = (cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000.)[-1]
            best_sharpe_medium_series = (cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000.).copy()

            best_sharpe_recent_trial = itrial
            best_sharpe_recent_model = weights_filename
            best_sharpe_recent_ratio = sharpe_recent_system
            best_sharpe_recent_final_value = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.)[-1]
            best_sharpe_recent_series = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.).copy()

        if not first_trial and cumu_system[-1] > best_final_value:
            best_model = weights_filename
            best_final_value = cumu_system[-1]
            best_trial = itrial
            best_series = cumu_system.copy()
        if not first_trial and cumu_system[-1]/cumu_system[-num_months_medium]*10000. > best_medium_final_value:
            best_medium_model = weights_filename
            best_medium_final_value = cumu_system[-1]/cumu_system[-num_months_medium]*10000.
            best_medium_trial = itrial
            best_medium_series = (cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000.).copy()
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
        if not first_trial and sharpe_medium_system > best_sharpe_medium_ratio:
            best_sharpe_medium_model = weights_filename
            best_sharpe_medium_ratio = sharpe_medium_system
            best_sharpe_medium_final_value = (cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000.)[-1]
            best_sharpe_medium_trial = itrial
            best_sharpe_medium_series = (cumu_system[-num_months_medium:]/cumu_system[-num_months_medium]*10000.).copy()
        if not first_trial and sharpe_recent_system > best_sharpe_recent_ratio:
            best_sharpe_recent_model = weights_filename
            best_sharpe_recent_ratio = sharpe_recent_system
            best_sharpe_recent_final_value = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.)[-1]
            best_sharpe_recent_trial = itrial
            best_sharpe_recent_series = (cumu_system[-num_months:]/cumu_system[-num_months]*10000.).copy()

        print("\n\n ... Best model: trial = ", best_trial, " file = ", best_model)
        print(" ... Best final value = ", format(best_final_value, '10,.0f'))
        print("\n ... Best medium model: trial = ", best_medium_trial, " file = ", best_medium_model)
        print(" ... Best medium final value = ", format(best_medium_final_value, '10,.0f'), "\n\n\n")
        print("\n ... Best recent model: trial = ", best_recent_trial, " file = ", best_recent_model)
        print(" ... Best recent final value = ", format(best_recent_final_value, '10,.0f'), "\n\n\n")

        print("\n\n ... Best sharpe model: trial = ", best_sharpe_trial, " file = ", best_sharpe_model)
        print(" ... Best sharpe ratio = ", format(best_sharpe_ratio, '6.3f'))
        print(" ... Best sharpe value = ", format(best_sharpe_final_value, '10,.0f'))
        print("\n ... Best sharpe medium model: trial = ", best_sharpe_medium_trial, " file = ", best_sharpe_medium_model)
        print(" ... Best sharpe medium_ratio = ", format(best_sharpe_medium_ratio, '6.3f'))
        print(" ... Best sharpe_medium value = ", format(best_sharpe_medium_final_value, '10,.0f'), "\n\n\n")
        print("\n ... Best sharpe recent model: trial = ", best_sharpe_recent_trial, " file = ", best_sharpe_recent_model)
        print(" ... Best sharpe recent_ratio = ", format(best_sharpe_recent_ratio, '6.3f'))
        print(" ... Best sharpe_recent value = ", format(best_sharpe_recent_final_value, '10,.0f'), "\n\n\n")

        print("\ntraining history = ", history.history['val_loss'][-1], history.history['loss'][-1], "\n\n\n")

        # update file with training parameters
        _validation_loss = np.min(history.history['val_loss'])
        saved_model_index = np.argmin(history.history['val_loss'])
        _loss = list(history.history['loss'])[saved_model_index]
        elapsed_time = (time.time() - start_time) / 60.
        with open(params_filename, 'a') as f:
            f.write("_validation_loss: "+str(_validation_loss)+"\n")
            f.write("_loss: "+str(_loss)+"\n")
            f.write("_final_value_BH: "+format(cumu_BH[-1], '10,.0f')+"\n")
            f.write("_final_value_system: "+format(cumu_system[-1], '10,.0f')+"\n")
            f.write("_final_recent_value_BH: "+format(cumu_BH[-1]/cumu_BH[-num_months]*10000., '10,.0f')+"\n")
            f.write("_final_medium_value_system: "+format(cumu_system[-1]/cumu_system[-num_months_medium]*10000., '10,.0f')+"\n")
            f.write("_final_recent_value_system: "+format(cumu_system[-1]/cumu_system[-num_months]*10000., '10,.0f')+"\n")
            f.write("_sharpe_ratio_BH: "+str(sharpe_BH)+"\n")
            f.write("_sharpe_ratio_system: "+str(sharpe_system)+"\n")
            f.write("_sharpe_ratio_system_worst: "+str(sharpe_system_worst)+"\n")
            f.write("_sharpe_ratio_medium_BH: "+str(sharpe_recent_BH)+"\n")
            f.write("_sharpe_ratio_medium_system: "+str(sharpe_medium_system)+"\n")
            f.write("_sharpe_ratio_medium_system_worst: "+str(sharpe_medium_system_worst)+"\n")
            f.write("_sharpe_ratio_recent_BH: "+str(sharpe_recent_BH)+"\n")
            f.write("_sharpe_ratio_recent_system: "+str(sharpe_recent_system)+"\n")
            f.write("_sharpe_ratio_recent_system_worst: "+str(sharpe_recent_system_worst)+"\n")
            f.write("elapsed_time (minutes): "+str(elapsed_time)+"\n")

        if first_trial:
            # print header for csv file
            datestamp = time.strftime("_%Y-%m-%d")
            csv_filename = "pngs/"+str(datestamp)+'.csv'
            with open(csv_filename,'a') as f:
                f.write("itrial,"+
                        "timestamp,"+
                        "stockList,"+
                        "num_stocks,"+
                        "first_history_index,"+
                        "shortest_incr,"+
                        "long_incr,"+
                        "incr_incr,"+
                        "num_periods_history,"+
                        "missing_years,"+
                        "missing_stocks_indices,"+
                        "missing_stocks,"+
                        "DL_channels,"+
                        "number_feature_maps,"+
                        "feature_map_factor,"+
                        "perform_batch_normalization,"+
                        "dense_factor,"+
                        "use_leaky_relu,"+
                        "leaky_relu_alpha,"+
                        "dropout_pct,"+
                        "optimizer_choice,"+
                        "learning_rate,"+
                        "stockList_predict,"+
                        "weights_filename,"+
                        "model_json_filename,"+
                        "_loss,"+
                        "_final_value_BH,"+
                        "_final_medium_value_BH,"+
                        "_final_recent_value_BH,"+
                        "_sharpe_ratio_BH,"+
                        "_sharpe_ratio_medium_BH,"+
                        "_sharpe_ratio_recent_BH,"+
                        "_validation_loss,"+
                        "_final_value_system,"+
                        "_final_medium_value_system,"+
                        "_final_recent_value_system,"+
                        "_sharpe_ratio_system,"+
                        "_sharpe_ratio_medium_system,"+
                        "_sharpe_ratio_recent_system,"+
                        "_sharpe_ratio_system_worst,"+
                        "_sharpe_ratio_medium_system_worst,"+
                        "_sharpe_ratio_recent_system_worst,"+
                        "elapsed_time"+"\n"
                        )

        with open(csv_filename,'a') as f:
            f.write(str(itrial)+", "+\
                    timestamp+", "+\
                    stockList+", "+\
                    str(num_stocks)+", "+\
                    str(first_history_index)+", "+\
                    str(shortest_incr)+", "+\
                    str(long_incr)+", "+\
                    str(incr_incr)+", "+\
                    str(num_periods_history)+", "+\
                    str(missing_years).replace(",",";")+", "+\
                    str(list(missing_stocks)).replace(",",";")+", "+\
                    str(list(np.array(symbols)[missing_stocks])).replace(",",";")+", "+\
                    str(Xtrain.shape[-1])+", "+\
                    str(number_feature_maps)+", "+\
                    str(feature_map_factor)+", "+\
                    str(perform_batch_normalization)+", "+\
                    str(dense_factor)+", "+\
                    str(use_leaky_relu)+", "+\
                    str(leaky_relu_alpha)+", "+\
                    str(dropout_pct)+", "+\
                    optimizer_choice+", "+\
                    str(learning_rate)+", "+\
                    loss_function+", "+\
                    stockList_predict+", "+\
                    weights_filename+", "+\
                    model_json_filename+", "+\
                    str(_loss)+", "+\
                    format(cumu_BH[-1], '10.0f')+", "+\
                    format(cumu_BH[-1]/cumu_BH[-num_months_medium]*10000., '10.0f')+", "+\
                    format(cumu_BH[-1]/cumu_BH[-num_months]*10000., '10.0f')+", "+\
                    str(sharpe_BH)+", "+\
                    str(sharpe_medium_BH)+", "+\
                    str(sharpe_recent_BH)+", "+\
                    str(_validation_loss)+", "+\
                    format(cumu_system[-1], '10.0f')+", "+\
                    format(cumu_system[-1]/cumu_system[-num_months_medium]*10000., '10.0f')+", "+\
                    format(cumu_system[-1]/cumu_system[-num_months]*10000., '10.0f')+", "+\
                    str(sharpe_system)+", "+\
                    str(sharpe_medium_system)+", "+\
                    str(sharpe_recent_system)+", "+\
                    str(sharpe_system_worst)+", "+\
                    str(sharpe_medium_system_worst)+", "+\
                    str(sharpe_recent_system_worst)+", "+\
                    str(elapsed_time)+"\n"
                    )

        first_trial = False

        plt.close('all')
        del model
        K.clear_session()




