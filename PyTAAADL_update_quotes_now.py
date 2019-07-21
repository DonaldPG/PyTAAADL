# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:27:52 2019

@author: Don
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:00:11 2019

@author: Don
"""

import datetime
import os
import time

from functions.GetParams import GetParams, GetHoldings
from functions.UpdateSymbols_inHDF5 import UpdateHDF_yf
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.readSymbols import get_Naz100List, get_SP500List

import sys
#print(sys.path)

def _main():

    # Get Credentials for sending email
    params = GetParams()
    print("")
    print("params = ", params)
    print("")
    stockList = params['stockList_predict']

    print("\n\n\n*************************************************************")
    print("... top of _main in PyTAAA.py ...")
    print("*************************************************************")

    # Update prices in HDF5 file for symbols in list
    # - check web for current stocks in Naz100, update files if changes needed
    try:
        if stockList == 'Naz100':
            _, removedTickers, addedTickers = get_Naz100List(verbose=True)
        elif stockList == 'SP500':
            _, removedTickers, addedTickers = get_SP500List(verbose=True)
    except:
        removedTickers, addedTickers = [], []

    symbol_directory = os.path.join(os.getcwd(), "symbols")

    if stockList == 'Naz100':
        symbol_file = "Naz100_Symbols.txt"
    elif stockList == 'SP500':
        symbol_file = "SP500_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )

    start_time = time.time()
    print("...start UpdateHDF_yf...")
    print(" ... symbol_directory, symbols_file = ", symbol_directory, symbols_file)
    UpdateHDF_yf( symbol_directory, symbols_file)
    print("...finish UpdateHDF_yf...")
    elapsed_time = time.time() - start_time
    print("elapsed time for quote updates: ", format(elapsed_time,'6.1f'), " seconds")

    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(symbols_file)
    lastdate = datearray[-1]


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    _main()