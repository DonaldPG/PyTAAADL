# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:37:39 2019

@author: Don
"""

import os
import numpy as np
import datetime
import pandas as pd

def create_minimal_hdf(dirname, listname='Naz100'):

    def _return_quotes_array( symbols_file, start_date="2018-01-01", end_date=None ):
        ###
        ### get quotes from yahoo_fix. return quotes, symbols, dates
        ### as numpy arrays
        ###
        import datetime
        from functions.readSymbols import readSymbolList
        from pandas_datareader import data as pdr
        import functions.fix_yahoo_finance as yf
        yf.pdr_override() # <== that's all it takes :-)

        # read symbols list
        symbols = readSymbolList(symbols_file,verbose=True)

        if end_date == None:
            end_date = str(datetime.datetime.now()).split(' ')[0]
        data = pdr.get_data_yahoo(symbols, start=start_date, end=end_date)
        try:
            # for multiple symbols
            symbolList = data['Adj Close'].columns
        except:
            # for single symbol
            symbolList = symbols
        datearray = data['Adj Close'].index
        x = data['Adj Close'].values
        newdates = []
        for i in range(datearray.shape[0]):
            newdates.append(str(datearray[i]).split(' ')[0])
        newdates = np.array(newdates)

        if x.ndim==1:
            x = x.reshape(x.size, 1)

        return x, symbolList, newdates

    ##
    ## Get quotes for the new symbol
    ##

    new_symbols = ['MSFT']
    symbol_directory = os.path.join(dirname, 'symbols')

    # write new symbols to temporary file
    if len(new_symbols) > 0:
        # write new symbols to temporary file
        symbols_file = os.path.join(symbol_directory, "newsymbols_tempfile.txt")
        OUTFILE = open(symbols_file,"w")
        for i,isymbol in enumerate(new_symbols):
            print("new symbol = ", isymbol)
            OUTFILE.write(str(isymbol) + "\n")
        OUTFILE.close()

        newquotesfirstdate = datetime.date(1991,1,1)
        newquoteslastdate = datetime.date(1991,1,7)

        # print dates to be used
        print("dates for new symbol found = ", newquotesfirstdate, newquoteslastdate)
        print("newquotesfirstdate, newquoteslastdate = ", newquotesfirstdate, newquoteslastdate)

        newadjClose, newsymbols, newdatearray = _return_quotes_array(symbols_file,
                                                    start_date=newquotesfirstdate,
                                                    end_date=newquoteslastdate)

        if type(newdatearray) == list:
            newdatearray = np.array(newdatearray)
        print(" newadjClose.shape = ", newadjClose.shape)
        print(" len(newsymbols) = ", len(newsymbols))
        print(" len(newdatearray) = ", len(newdatearray))
        print(" security values check: ",newadjClose[np.isnan(newadjClose)].shape)

        newdates = []
        for i in range(newdatearray.shape[0]):
            newdates.append(str(newdatearray[i]))
        print("newadjClose.shape = ", newadjClose)
        print('newsymbols = ', newsymbols)
        print('newdatearray = ', newdatearray)
        if newadjClose.shape[1] == len(newdates):
            quotes_NewSymbols = pd.DataFrame(newadjClose.swapaxes(0,1), index=newdates, columns=newsymbols)
        else:
            quotes_NewSymbols = pd.DataFrame(newadjClose, index=newdates, columns=newsymbols)

    # set up to write quotes to disk.
    listname = 'Naz100' + "_Symbols"

    hdf5filename = os.path.join( symbol_directory, listname + "_.hdf5" )
    print("hdf5 filename = ",hdf5filename)
    quotes_NewSymbols.to_hdf( hdf5filename, listname, mode='a',format='table',append=False,complevel=5,complib='blosc')

    stock_list_filename = os.path.join(symbol_directory, listname + ".txt")
    import shutil
    shutil.copyfile(symbols_file, stock_list_filename)

    return


#---------------------------------------------
# Re-create the hdf5 file used to store stock quotes
#---------------------------------------------

# set up to write quotes to disk.
# - listname should be one of : 'Naz100', 'SP500', 'RU1000'
dirname = os.path.dirname(os.path.abspath(__file__))
listname = "Naz100"

create_minimal_hdf(dirname, listname='Naz100')
