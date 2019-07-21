import numpy as np
import datetime

#import datetime
#from scipy import random
#from scipy.stats import rankdata

#import functions.quotes_adjClose
from functions.quotes_adjClose import downloadQuotes
#from functions.TAfunctions import *
from functions.readSymbols import readSymbolList

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
    quote = quote.convert_objects(convert_numeric=True)   ### test
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
    " use alpha_vantage instead of google "
    from .quotes_adjClose_alphavantage import get_last_quote
    if symbol == 'CASH':
        last_quote = 1.0
    else:
        last_quote = get_last_quote(symbol)
    return last_quote


def get_quote_alphavantage( symbol ):
    " use alpha_vantage instead of google "
    try:
        from functions.quotes_adjClose_alphavantage import get_last_quote
        from functions.quotes_adjClose_alphavantage import get_last_quote_daily
    except:
        from .quotes_adjClose_alphavantage import get_last_quote
        from .quotes_adjClose_alphavantage import get_last_quote_daily
    try:
        _quote = get_last_quote(symbol)
    except:
        _quote = get_last_quote_daily(symbol)
    return _quote


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


def get_pe_finviz( symbol, verbose=False ):
    ' use finviz to get calculated P/E ratios '
    import numpy as np
    import bs4 as bs
    import requests
    import os
    import csv

    try:
        # Get source table
        url = 'https://finviz.com/quote.ashx?t='+symbol.upper()
        r = requests.get(url)
        html = r .text
        soup = bs.BeautifulSoup(html, 'lxml')
        table = soup.find('table', class_= 'snapshot-table2')

        # Split by row and extract values
        values = []
        for tr in table.find_all('tr')[1:3]:
            td = tr.find_all('td')[1]
            value = td.text

            #Convert to numeric
            if 'B' in value:
                value = value.replace('B',"")
                value = float(value.strip())
                value = value * 1000000000
                values.append(value)

            elif 'M' in value:
                value = value.replace('M',"")
                value = float(value.strip())
                value = value * 1000000
                values.append(value)

            #Account for blank values
            else:
                values.append(0)

        #Append to respective lists
        market_cap = values[0]
        earnings = values[1]
        if float(earnings) != 0.:
            pe = market_cap / earnings
        else:
            pe = np.nan
    except:
        market_cap = 0.
        earnings = 0.
        pe = np.nan
    if verbose:
        print((symbol+' market cap, earnings, P/E = '+str(market_cap)+', '+str(earnings)+', '+str(pe)))
    return pe


def get_SectorAndIndustry_google( symbol ):
    import urllib.request, urllib.parse, urllib.error
    import re
    base_url = 'http://finance.google.com/finance?q=NASDAQ%3A'
    content = urllib.request.urlopen(base_url + symbol).read()
    try:
        m = content.split("Sector:")[1].split('>')[1].split("<")[0].replace("&amp;","and")
        sector = m
    except:
        sector = ""
    try:
        m = content.split("Industry:")[1].split('>')[1].split("<")[0].replace("&amp;","and").replace(" - NEC","")
        industry = m
    except:
        industry = ""
    return sector, industry


def get_SectorAndIndustry_google( symbol ):
    ' use finviz to get sector and industry '
    import requests
    import bs4 as bs
    import os
    import csv
    #Get source table
    url = 'https://finviz.com/quote.ashx?t='+symbol.upper()
    r = requests.get(url)
    html = r .text
    soup = bs.BeautifulSoup(html, 'lxml')
    table = soup.find('table', class_= 'fullview-title')

    for tr in table.find_all('tr')[2:3]:
        td = tr.find_all('td')[0]
        value = td.text
        print(value)
    values = value.split(' | ')[:-1]
    #print values
    industry = values[0]
    sector = values[1]
    return sector, industry


def LastQuotesForSymbolList( symbolList ):
    """
    read in latest (15-minute delayed) quote for each symbol in list.
    Use alpha_vantage for each symbol's quote.
    """
    from time import sleep
    #from functions.quote_adjClose_alphavantage import get_last_quote

    def scrape_quote(_symbol):
        quote = get_quote_alphavantage( _symbol )
        #print "ticker, quote = ", _symbol, quote
        # Remove comma from quote
        if type(quote) == 'str' and ',' in quote:
            quote = quote.replace(",", "")
        return quote

    quotelist = []
    for itick, ticker in enumerate( symbolList ):

        if ticker == 'CASH':
            print("ticker, quote = CASH 1.0")
            quotelist.append(1.0)
        else:
            for i in range(200):
                try:
                    quote = scrape_quote(ticker)
                    break
                except:
                    print((".",))
                    quote = scrape_quote(ticker)
            print(("ticker, quote = ", ticker, quote))
            quotelist.append( quote )

    return quotelist


def LastQuotesForSymbolList_hdf( symbolList, symbols_file ):
    """
    read in latest (15-minute delayed) quote for each symbol in list.
    Use quotes on hdf for each symbol's quote.
    """
    from .UpdateSymbols_inHDF5 import loadQuotes_fromHDF
    # get last quote in pyTAAA hdf file
    _, _, _, quote, _ = loadQuotes_fromHDF( symbols_file )
    quotelist = []
    for itick, ticker in enumerate( symbolList ):
        quotelist.append(quote[ticker].values[-1])
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
