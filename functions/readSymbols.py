'''
Created on May 12, 202

@author: donaldpg
'''
import os
from functions.quotes_for_list_adjClose import *
        
def readSymbolList(filename,verbose=False):
    # Get the Data
    try:
        print(" ...inside readSymbolList... filename = ", filename)
        infile = file(filename,"r")
    except:
        
        symbol_directory = os.path.join( os.getcwd(), "symbols" )
        # the symbols list doesn't exist. generate from the web.
        if 'SP500' in filename:
            symbol_file = "SP500_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_SP500List( verbose=True )
            infile.close()
            infile = file(filename,"r")
        elif 'Naz100' in filename:
            symbol_file = "Naz100_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_Naz100List( verbose=True )

    symbols = []

    content = infile.read()
    number_lines = len(content.split("\n"))
    if number_lines == 0:
        import os
        
        symbol_directory = os.path.join( os.getcwd(), "symbols" )
        # the symbols list doesn't exist. generate from the web.
        if 'SP500' in filename:
            symbol_file = "SP500_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_SP500List( verbose=True )
        elif 'Naz100' in filename:
            symbol_file = "Naz100_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_Naz100List( verbose=True )

    infile.close()
    infile = file(filename,"r")

    while infile:
        line = infile.readline()
        s = line.split()
        n = len(s)
        if n != 0:
            for i in range(len(s)):
                symbols.append(s[i])
        else:
            break

    # ensure that there are no duplicate tickers
    symbols = list( set( symbols ) )

    # sort list of symbols
    symbols.sort()

    # print list of symbols
    if verbose:
        print("number of symbols is ",len(symbols))
        print(symbols)

    return symbols
