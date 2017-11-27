import numpy as np
import datetime

import datetime
from scipy import random
from scipy.stats import rankdata

import functions.quotes_adjClose
from functions.quotes_adjClose import *
from functions.TAfunctions import *
from functions.readSymbols import *

"""
def get_SP500List( verbose=True ):
    ###
    ### Query wikipedia.com for updated list of stocks in S&P 500 index.
    ### Return list with stock tickers.
    ###
    import urllib2
    import os
    import datetime
    from bs4 import BeautifulSoup

    ###
    ### get symbol list from previous period
    ###
    symbol_directory = os.path.join( os.getcwd(), "symbols" )

    symbol_file = "SP500_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    with open(symbols_file, "r+") as f:
        old_symbolList = f.readlines()
    for i in range( len(old_symbolList) ) :
        old_symbolList[i] = old_symbolList[i].replace("\n","")

    ###
    ### get current symbol list from wikipedia website
    ###
    try:
        base_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        #content = urllib2.urlopen(base_url)
        #c = content.read()
        #print "\n\n\n content = ", content
        #print "... got web content"

        soup = BeautifulSoup( urllib2.urlopen(base_url) )
        #soup = BeautifulSoup(content.read())
        t = soup.find("table", {"class" : "wikitable sortable"})

        print "... got web content"
        print "... ran beautiful soup on web content"

        symbolList = [] # store all of the records in this list
        companyNamesList = []
        industry = []
        subIndustry = []
        for row in t.findAll('tr'):
            try:
                #print "\n\nrow = \n", row
                col = row.findAll('td')
                _ticker = col[0].string.strip()
                _company = col[1].string.strip()
                _sector = col[3].string.strip()
                _subIndustry = col[4].string.strip()
                symbolList.append(_ticker)
                companyNamesList.append(_company)
                industry.append(_sector)
                subIndustry.append(_subIndustry)
            except:
                pass
        print "... retrieved SP500 companies lists from internet"

        companyName_file = os.path.join( symbol_directory, "SP500_companyNames.txt" )
        with open( companyName_file, "w" ) as f:
            for i in range( len(symbolList) ) :
                f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )
        print "... wrote SP500_companyNames.txt"

        ###
        ### compare old list with new list and print changes, if any
        ###

        # file for index changes history
        symbol_change_file = "SP500_symbolsChanges.txt"
        symbols_changes_file = os.path.join( symbol_directory, symbol_change_file )
        if not os.path.isfile(symbols_changes_file):
            open(symbols_changes_file, 'a').close()
        with open(symbols_changes_file, "r+") as f:
            old_symbol_changesList = f.readlines()
        old_symbol_changesListText = ''
        for i in range( len(old_symbol_changesList) ):
            old_symbol_changesListText = old_symbol_changesListText + old_symbol_changesList[i]

        # parse date
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        day = datetime.datetime.now().day
        dateToday = str(year)+"-"+str(month)+"-"+str(day)

        # compare lists to check for tickers removed from the index
        # - printing will be suppressed if "verbose = False"
        removedTickers = []
        print ""
        for i, ticker in enumerate( old_symbolList ):
            if i == 0:
                removedTickersText = ''
            if ticker not in symbolList:
                removedTickers.append( ticker )
                if verbose:
                    print " Ticker ", ticker, " has been removed from the SP500 index"
                removedTickersText = removedTickersText + "\n" + dateToday + " Remove " + ticker

        # compare lists to check for tickers added to the index
        # - printing will be suppressed if "verbose = False"
        addedTickers = []
        print ""
        for i, ticker in enumerate( symbolList ):
            if i == 0:
                addedTickersText = ''
            if ticker not in old_symbolList:
                addedTickers.append( ticker )
                if verbose:
                    print " Ticker ", ticker, " has been added to the SP500 index"
                addedTickersText = addedTickersText + "\n" + dateToday + " Add    " + ticker

        print ""
        with open(symbols_changes_file, "w") as f:
            f.write(addedTickersText)
            f.write(removedTickersText)
            f.write("\n")
            f.write(old_symbol_changesListText)

        print "****************"
        print "addedTickers = ", addedTickers
        print "removedTickers = ", removedTickers
        print "****************"
        ###
        ### update symbols file with current list. Keep copy of of list.
        ###

        if removedTickers != [] or addedTickers != []:

            # make copy of previous symbols list file
            symbol_directory = os.path.join( os.getcwd(), "symbols" )
            symbol_file = "SP500_Symbols.txt"
            archive_symbol_file = "SP500_symbols__" + str(datetime.date.today()) + ".txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

            with open( archive_symbols_file, "w" ) as f:
                for i in range( len(old_symbolList) ) :
                    f.write( old_symbolList[i] + "\n" )

            # make new symbols list file
            with open( symbols_file, "w" ) as f:
                for i in range( len(symbolList) ) :
                    f.write( symbolList[i] + "\n" )
    except:
        ###
        ### something didn't wor. print message and return old list.
        ###
        print "\n\n\n"
        print "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! "
        print " SP500 sysmbols list did not get updated from web."
        print " ... check quotes_for_list_adjCloseVol.py in function 'get_SP500List' "
        print " ... also check web at en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        print "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! "
        print "\n\n\n"

        symbolList = old_symbolList
        removedTickers = []
        addedTickers = []

    return symbolList, removedTickers, addedTickers
"""

def get_SP500List( verbose=True ):
    ###
    ### Query wikipedia.com for updated list of stocks in S&P 500 index.
    ### Return list with stock tickers.
    ###
    import urllib.request, urllib.error, urllib.parse
    import os
    import datetime
    from bs4 import BeautifulSoup

    ###
    ### get symbol list from previous period
    ###
    symbol_directory = os.path.join( os.getcwd(), "symbols" )

    symbol_file = "SP500_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    with open(symbols_file, "r+") as f:
        old_symbolList = f.readlines()
    for i in range( len(old_symbolList) ) :
        old_symbolList[i] = old_symbolList[i].replace("\n","")

    ###
    ### get current symbol list from wikipedia website
    ###

    base_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    #content = urllib2.urlopen(base_url)
    #c = content.read()
    #print "\n\n\n content = ", content
    #print "... got web content"

    soup = BeautifulSoup( urllib.request.urlopen(base_url) )
    #soup = BeautifulSoup(content.read())
    t = soup.find("table", {"class" : "wikitable sortable"})

    print("... got web content")
    print("... ran beautiful soup on web content")

    symbolList = [] # store all of the records in this list
    companyNamesList = []
    industry = []
    subIndustry = []
    for row in t.findAll('tr'):
        try:
            #print "\n\nrow = \n", row
            col = row.findAll('td')
            _ticker = col[0].string.strip()
            _company = col[1].string.strip()
            _sector = col[3].string.strip()
            _subIndustry = col[4].string.strip()
            symbolList.append(_ticker)
            companyNamesList.append(_company)
            industry.append(_sector)
            subIndustry.append(_subIndustry)
        except:
            pass
    print("... retrieved SP500 companies lists from internet")

    companyName_file = os.path.join( symbol_directory, "SP500_companyNames.txt" )
    with open( companyName_file, "w" ) as f:
        for i in range( len(symbolList) ) :
            f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )
    print("... wrote SP500_companyNames.txt")

    ###
    ### compare old list with new list and print changes, if any
    ###

    # file for index changes history
    symbol_change_file = "SP500_symbolsChanges.txt"
    symbols_changes_file = os.path.join( symbol_directory, symbol_change_file )
    if not os.path.isfile(symbols_changes_file):
        open(symbols_changes_file, 'a').close()
    with open(symbols_changes_file, "r+") as f:
        old_symbol_changesList = f.readlines()
    old_symbol_changesListText = ''
    for i in range( len(old_symbol_changesList) ):
        old_symbol_changesListText = old_symbol_changesListText + old_symbol_changesList[i]

    # parse date
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    dateToday = str(year)+"-"+str(month)+"-"+str(day)

    # compare lists to check for tickers removed from the index
    # - printing will be suppressed if "verbose = False"
    removedTickers = []
    removedTickersText = ''
    print("")
    for i, ticker in enumerate( old_symbolList ):
        if i == 0:
            removedTickersText = ''
        if ticker not in symbolList:
            removedTickers.append( ticker )
            if verbose:
                print((" Ticker ", ticker, " has been removed from the SP500 index"))
            removedTickersText = removedTickersText + "\n" + dateToday + " Remove " + ticker

    # compare lists to check for tickers added to the index
    # - printing will be suppressed if "verbose = False"
    addedTickers = []
    print("")
    for i, ticker in enumerate( symbolList ):
        if i == 0:
            addedTickersText = ''
        if ticker not in old_symbolList:
            addedTickers.append( ticker )
            if verbose:
                print((" Ticker ", ticker, " has been added to the SP500 index"))
            addedTickersText = addedTickersText + "\n" + dateToday + " Add    " + ticker

    print("")
    with open(symbols_changes_file, "w") as f:
        f.write(addedTickersText)
        f.write(removedTickersText)
        f.write("\n")
        f.write(old_symbol_changesListText)

    print("****************")
    print(("addedTickers = ", addedTickers))
    print(("removedTickers = ", removedTickers))
    print("****************")
    ###
    ### update symbols file with current list. Keep copy of of list.
    ###

    if removedTickers != [] or addedTickers != []:

        # make copy of previous symbols list file
        symbol_directory = os.path.join( os.getcwd(), "symbols" )
        symbol_file = "SP500_Symbols.txt"
        archive_symbol_file = "SP500_symbols__" + str(datetime.date.today()) + ".txt"
        symbols_file = os.path.join( symbol_directory, symbol_file )
        archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

        with open( archive_symbols_file, "w" ) as f:
            for i in range( len(old_symbolList) ) :
                f.write( old_symbolList[i] + "\n" )

        # make new symbols list file
        with open( symbols_file, "w" ) as f:
            for i in range( len(symbolList) ) :
                f.write( symbolList[i] + "\n" )

    return symbolList, removedTickers, addedTickers



def get_Naz100List( verbose=True ):
    ###
    ### Query nasdaq.com for updated list of stocks in Nasdaq 100 index.
    ### Return list with stock tickers.
    ###
    #import urllib
    import requests
    import re
    import os
    import datetime

    ###
    ### get symbol list from previous period
    ###
    symbol_directory = os.path.join( os.getcwd(), "symbols" )

    symbol_file = "Naz100_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    with open(symbols_file, "r+") as f:
        old_symbolList = f.readlines()
    for i in range( len(old_symbolList) ) :
        old_symbolList[i] = old_symbolList[i].replace("\n","")

    ###
    ### get current symbol list from nasdaq website
    ###
    try:
        base_url = 'http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx'
        content = requests.get(base_url).text
        #print "\n\n\n content = ", content

        m = re.search('var table_body.*?>*(?s)(.*?)<.*?>.*?<', content).group(0).split("],[")
        # handle exceptions in format for first and last entries in list
        m = m[0].split(",\r\n")
        m[0] = m[0].split("[")[2]
        m[-1] = m[-1].split("];")[0]
        print("****************")
        for i in range( len(m) ):
            print((i, m[i]))
        print(("len of m = ",len(m)))
        print("****************")
        # parse list items for symbol name
        symbolList = []
        companyNamesList = []
        for i in range( len(m) ):
            symbolList.append( m[i].split(",")[0].split('"')[1] )
            companyNamesList.append( m[i].split(",")[1].split('"')[1] )

        companyName_file = os.path.join( symbol_directory, "companyNames.txt" )
        with open( companyName_file, "w" ) as f:
            for i in range( len(symbolList) ) :
                f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )

        ###
        ### compare old list with new list and print changes, if any
        ###

        # file for index changes history
        symbol_change_file = "Naz100_symbolsChanges.txt"
        symbols_changes_file = os.path.join( symbol_directory, symbol_change_file )
        with open(symbols_changes_file, "r+") as f:
            old_symbol_changesList = f.readlines()
        old_symbol_changesListText = ''
        for i in range( len(old_symbol_changesList) ):
            old_symbol_changesListText = old_symbol_changesListText + old_symbol_changesList[i]

        # parse date
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        day = datetime.datetime.now().day
        dateToday = str(year)+"-"+str(month)+"-"+str(day)

        # compare lists to check for tickers removed from the index
        # - printing will be suppressed if "verbose = False"
        removedTickers = []
        print("")
        for i, ticker in enumerate( old_symbolList ):
            if i == 0:
                removedTickersText = ''
            if ticker not in symbolList:
                removedTickers.append( ticker )
                if verbose:
                    print((" Ticker ", ticker, " has been removed from the Nasdaq100 index"))
                removedTickersText = removedTickersText + "\n" + dateToday + " Remove " + ticker

        # compare lists to check for tickers added to the index
        # - printing will be suppressed if "verbose = False"
        addedTickers = []
        print("")
        for i, ticker in enumerate( symbolList ):
            if i == 0:
                addedTickersText = ''
            if ticker not in old_symbolList:
                addedTickers.append( ticker )
                if verbose:
                    print((" Ticker ", ticker, " has been added to the Nasdaq100 index"))
                addedTickersText = addedTickersText + "\n" + dateToday + " Add    " + ticker

        print("")
        with open(symbols_changes_file, "w") as f:
            f.write(addedTickersText)
            f.write(removedTickersText)
            f.write("\n")
            f.write(old_symbol_changesListText)

        print("****************")
        print(("addedTickers = ", addedTickers))
        print(("removedTickers = ", removedTickers))
        print("****************")
        ###
        ### update symbols file with current list. Keep copy of of list.
        ###

        if removedTickers != [] or addedTickers != []:

            # make copy of previous symbols list file
            symbol_directory = os.path.join( os.getcwd(), "symbols" )
            symbol_file = "Naz100_Symbols.txt"
            archive_symbol_file = "Naz100_symbols__" + str(datetime.date.today()) + ".txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

            with open( archive_symbols_file, "w" ) as f:
                for i in range( len(old_symbolList) ) :
                    f.write( old_symbolList[i] + "\n" )

            # make new symbols list file
            with open( symbols_file, "w" ) as f:
                for i in range( len(symbolList) ) :
                    f.write( symbolList[i] + "\n" )
    except:
        ###
        ### something didn't wor. print message and return old list.
        ###
        print("\n\n\n")
        print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
        print(" Nasdaq sysmbols list did not get updated from web.")
        print(" ... check quotes_for_list_adjCloseVol.py in function 'get_Naz100List' ")
        print(" ... also check web at http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx")
        print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
        print("\n\n\n")

        symbolList = old_symbolList
        removedTickers = []
        addedTickers = []


    return symbolList, removedTickers, addedTickers


def get_Naz100PlusETFsList( verbose=True ):
    ###
    ### Query nasdaq.com for updated list of stocks in Nasdaq 100 index.
    ### Return list with stock tickers.
    ###
    import urllib.request, urllib.parse, urllib.error
    import re
    import os
    import datetime

    ###
    ### get symbol list from previous period
    ###
    symbol_directory = os.path.join( os.getcwd(), "symbols" )
    symbol_file = "Naz100PlusETFs_symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    with open(symbols_file, "r+") as f:
        old_symbolList = f.readlines()
    for i in range( len(old_symbolList) ) :
        old_symbolList[i] = old_symbolList[i].replace("\n","")

    ###
    ### get current symbol list from nasdaq website
    ###
    base_url = 'http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx'
    content = urllib.request.urlopen(base_url).read()
    m = re.search('var table_body.*?>*(?s)(.*?)<.*?>.*?<', content).group(0).split("],[")
    # handle exceptions in format for first and last entries in list
    m[0] = m[0].split("[")[2]
    m[-1] = m[-1].split("]")[0].split("[")[0]
    # parse list items for symbol name
    symbolList = []
    for i in range( len(m) ):
        symbolList.append( m[i].split(",")[0].split('"')[1] )

    ###
    ### compare old list with new list and print changes, if any
    ###

    # compare lists to check for tickers removed from the index
    # - printing will be suppressed if "verbose = False"
    removedTickers = []
    print("")
    for i, ticker in enumerate( symbolList ):
        if ticker not in old_symbolList:
            removedTickers.append( ticker )
            if verbose:
                print((" Ticker ", ticker, " has been removed from the Nasdaq100 index"))

    # add GTAA asset classes to Naz100 tickers for extra diversity
    ETF_List = ['AGG', 'CEW', 'DBC', 'EEM', 'EMB', 'FXE', 'GLD', 'HYG', 'IVV', 'LQD', 'TIP', 'TLT', 'USO', 'VNQ', 'XLF', 'XWD.TO' ]
    for i in range( len(ETF_List) ) :
        symbolList.append( ETF_List[i] )

    # compare lists to check for tickers added to the index
    # - printing will be suppressed if "verbose = False"
    addedTickers = []
    print("")
    for i, ticker in enumerate( old_symbolList ):
        if ticker not in symbolList:
            addedTickers.append( ticker )
            if verbose:
                print((" Ticker ", ticker, " has been added to the Nasdaq100 index"))

    print("")
    ###
    ### update symbols file with current list. Keep copy of of list.
    ###

    if removedTickers != [] or addedTickers != []:
        # make copy of previous symbols list file
        symbol_directory = os.path.join( os.getcwd(), "symbols" )
        symbol_file = "Naz100_Symbols.txt"
        archive_symbol_file = "Naz100_Symbols__" + str(datetime.date.today()) + ".txt"
        symbols_file = os.path.join( symbol_directory, symbol_file )
        archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

        with open( archive_symbols_file, "w" ) as f:
            for i in range( len(old_symbolList) ) :
                f.write( old_symbolList[i] + "\n" )

        # make new symbols list file
        with open( symbols_file, "w" ) as f:
            for i in range( len(symbolList) ) :
                f.write( symbolList[i] + "\n" )

    return symbolList.sort(), removedTickers, addedTickers


def arrayFromQuotesForList(symbolsFile, beginDate, endDate):
    '''
    read in quotes and process to 'clean' ndarray plus date array
    - prices in array with dimensions [num stocks : num days ]
    - process stock quotes to show closing prices adjusted for splits, dividends
    - single ndarray with dates common to all stocks [num days]
    - clean up stocks by:
       - infilling empty values with linear interpolated value
       - repeat first quote to beginning of series
    '''

    from functions.TAfunctions import interpolate
    from functions.TAfunctions import cleantobeginning

    # read symbols list
    symbols = readSymbolList(symbolsFile,verbose=True)

    # get quotes for each symbol in list (adjusted close)
    quote = downloadQuotes(symbols,date1=beginDate,date2=endDate,adjust=True,Verbose=True)

    # clean up quotes for missing values and varying starting date
    #x = quote.as_matrix().swapaxes(0,1)
    x = quote.values.T
    ###print "x = ", x
    date = quote.index
    date = [d.date().isoformat() for d in date]
    datearray = np.array(date)
    symbolList = list(quote.columns.values)

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    for ii in range(x.shape[0]):
        x[ii,:] = np.array(x[ii,:]).astype('float')
        #print " progress-- ", ii, " of ", x.shape[0], " symbol = ", symbols[ii]
        #print " line 283........."
        x[ii,:] = interpolate(x[ii,:])
        x[ii,:] = cleantobeginning(x[ii,:])

    return x, symbolList, datearray

def arrayFromQuotesForListWithVol(symbolsFile, beginDate, endDate):
    '''
    read in quotes and process to 'clean' ndarray plus date array
    - prices in array with dimensions [num stocks : num days ]
    - process stock quotes to show closing prices adjusted for splits, dividends
    - single ndarray with dates common to all stocks [num days]
    - clean up stocks by:
       - infilling empty values with linear interpolated value
       - repeat first quote to beginning of series
    '''

    # read symbols list
    symbols = readSymbolList(symbolsFile,verbose=True)

    # get quotes for each symbol in list (adjusted close)
    quote = downloadQuotes(symbols,date1=beginDate,date2=endDate,adjust=True,Verbose=True)

    # clean up quotes for missing values and varying starting date
    x=quote.copyx()
    x=quote.as_matrix().swapaxes(0,1)
    date = quote.getlabel(2)
    datearray = np.array(date)


    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    for ii in range(x.shape[0]):
        print(" line 315.........")
        x[ii,0,:] = interpolate(x[ii,0,:].values)
        x[ii,0,:] = cleantobeginning(x[ii,0,:].values)

    return x, symbolList, datearray


def get_quote_google( symbol ):
    import urllib.request, urllib.parse, urllib.error
    import re
    base_url = 'http://finance.google.com/finance?q=NASDAQ%3A'
    content = urllib.request.urlopen(base_url + symbol).read()
    m = re.search('class="pr".*?>*(?s)(.*?)<.*?>.*?<', content).group(0).split(">")[-1].split("<")[0]
    if m :
        quote = m
    else:
        quote = 'no quote available for: ' + symbol
    return quote

def get_pe_google( symbol ):
    import urllib.request, urllib.parse, urllib.error
    import re
    base_url = 'http://finance.google.com/finance?q=NASDAQ%3A'
    content = urllib.request.urlopen(base_url + symbol).read()
    try:
        m = float(content.split("pe_ratio")[1].split('\n')[2].split(">")[-1])
        quote = m
    except:
        quote = ""
    return quote

def LastQuotesForSymbolList( symbolList ):
    """
    read in latest (15-minute delayed) quote for each symbol in list.
    Use google for each symbol's quote.
    """
    from time import sleep
    quotelist = []
    for itick, ticker in enumerate( symbolList ):
        if ticker == 'CASH':
            print("ticker, quote = CASH 1.0")
            quotelist.append(1.0)
        else:
            try:
                data = get_quote_google( ticker )
                print(("ticker, quote = ", ticker, data))
                # Remove comma from data
                data = data.replace(",", "")
                quotelist.append( data )
            except:
                print(("could not get quote for ", ticker, "         will try again and again."))
                sleep(3)
                symbolList[itick+1:itick+1] = [ticker]
    return quotelist


def LastQuotesForList( symbols_list ):

    from time import sleep
    

    stocks = StockRetriever()

    # remove 'CASH' from symbols_list, if present. Keep track of position in list to re-insert
    cash_index = None
    try:
        cash_index = symbols_list.index('CASH')
        if cash_index >= 0 and cash_index <= len(symbols_list)-1 :
            symbols_list.remove('CASH')
    except:
        pass

    attempt = 1
    NeedQuotes = True
    while NeedQuotes:
        try:
            a=stocks.get_current_info( symbols_list )
            print(("inside LastQuotesForList 1, symbols_list = ", symbols_list))
            print(("inside LastQuotesForList 1, attempt = ", attempt))
            # convert from strings to numbers and put in a list
            quotelist = []
            for i in range(len(a)):
                singlequote = np.float((a[i]['LastTradePriceOnly']).encode('ascii','ignore'))
                quotelist.append(singlequote)
            print((symbols_list, quotelist))
            NeedQuotes = False
        except:
            attempt += 1
            sleep(attempt)

    print("inside LastQuotesForList... location  2")

    # re-insert CASH in original position and also add curent price of 1.0 to quotelist
    if cash_index != None:
        if cash_index < len(symbols_list):
            symbols_list[cash_index:cash_index] = 'CASH'
            quotelist[cash_index:cash_index] = 1.0
        else:
            symbols_list.append('CASH')
            quotelist.append(1.0)

    print(("attempts, sysmbols_list,quotelist =", attempt, symbols_list, quotelist))
    return quotelist
