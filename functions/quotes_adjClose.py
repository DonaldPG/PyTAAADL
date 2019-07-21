'''
Created on May 12, 202

@author: donaldpg
'''

def downloadQuotes(tickers, date1=None, date2=None, adjust=True, Verbose=False):
    """
    Given a ticker sequence, return historical Yahoo! quotes as a pandas DataFrame.

    Parameters
    ----------
    tickers : sequence
        A sequence (such as a list) of string tickers. For example:
        ['aapl', 'msft']
    date1 : {datetime.date, tuple}, optional
        The first date to grab historical quotes on. For example:
        datetime.date(2010, 1, 1) or (2010, 1, 1). By default the first
        date is (1900, 1, 1).
    date2 : {datetime.date, tuple}, optional
        The last date to grab historical quotes on. For example:
        datetime.date(2010, 12, 31) or (2010, 12, 31). By default the last
        date is 10 days beyond today's date.
    adjust : bool, optional
        Adjust (default) the open, close, high, and low prices. The
        adjustment takes splits and dividends into account such that the
        corresponding returns are correct. Volume is already split adjusted
        by Yahoo so it is not changed by the value of `adjust`.
    Verbose : bool, optional
        Print the ticker currently being loaded. By default the tickers are
        not printed.

    Returns
    -------
    quotes_df : DataFrame
        A pandas dataframe is returned. In order, the  axes contain: dates,
        quotes (adjusted close). The elements along the item axis depend on the value
        of `adjust`. When `adjust` is False, the items are

        ['open', 'close', 'high', 'low', 'volume', 'adjclose']

        When adjust is true (default), the adjusted close ('adjclose') is
        not included. The dates are datetime.date objects.

    Examples
    --------
    items = ['Adj Close']
    date1 = '2012-01-01'
    date2 = datetime.date.today()
    ticker = 'GOOGL'
    data = get_data_yahoo(ticker, start = date1, end = date2)[items]
    dates = data.index
    data.columns = [ticker]

    ticker = 'AMZN'
    data2 = get_data_yahoo(ticker, start = date1, end = date2)[items]
    dates2 = data2.index
    data2.columns = [ticker]

    data = data.join(data2, how='outer')
    data.sort_index( axis=0, inplace=True )

    data.tail()

                 GOOGL    AMZN
    Date
    2014-04-07  540.63  317.76
    2014-04-08  557.51  327.07
    2014-04-09  567.04  331.81
    2014-04-10  546.69  317.11
    2014-04-11  537.76  311.73


    """

    print(" ... inside quotes_adjClose/downloadQuotes ...")
    print("\ntickers = ", tickers)
    print("\ndate1 = ", date1)
    print("\ndate2 = ", date2)

    import datetime
    from time import sleep
    ##from matplotlib.finance import *
    ##from matplotlib.cbook import to_datetime
    #from la.external.matplotlib import quotes_historical_yahoo
    import pandas as pd
    #from pandas.io.data import DataReader
    #from pandas.io.data import get_data_yahoo, get_data_google
    ###from pandas_datareader.data import get_data_yahoo, get_data_google
    ###from pandas_datareader import get_data_yahoo, get_data_google
    #import la
    ####from functions.get_data_yahoo_fix import get_data_yahoo_fix  # <== that's all it takes :-)
    #from functions.quotes_adjClose_alphavantage import get_ts_data
    #from functions.quotes_adjClose_quandl import get_q_data

    if date1 is None:
        date1 = datetime.date(1900, 1, 1)
    if date2 is None:
        date2 = datetime.date.today() + datetime.timedelta(+2)
    #quotes_df = None
    #lar = None
    items = ['Adj Close']
    google_items = ['Close']
    if Verbose:
        print("Load data")

    i=0
    number_tries = 0
    re_tries = 0
    quandl_failure_count = 0
    for itick, ticker in enumerate(tickers):
        sleep(quandl_failure_count+1)
        if Verbose:
            print("\t" + ticker + "  ", end=' ')

        data = []
        dates = []

        #number_tries = 0
#        print " date1 = ", date1
#        print " date2 = ", date2
        date2 = datetime.date(date2.year,date2.month,date2.day)
#        print " date2 = ", date2
        if (date2 - date1).days <= 100:
            output_size = 'compact'
        else:
            output_size = 'full'
#        data, _ = get_ts_data(ticker, interval="D", outputsize=output_size, adjusted=True, output_format='df')
#        data = data[items]

        # read in dataframe containing adjusted close quotes for a ticker in the list
        #print "number_tries = ", number_tries
        if number_tries < 21:
            #print "number_tries = ", number_tries, " trying with yahoo"
            #data = get_data_yahoo_fix(ticker, start = date1, end = date2)[items]

            """
            try:
                ###data = get_data_yahoo(ticker, start = date1, end = date2)[items]
                ####data = get_data_yahoo_fix(ticker, start = date1, end = date2)[items]
                if (date2 - date1).days <= 100:
                    output_size = 'compact'
                else:
                    output_size = 'full'
                #print "   ... ticker = ", ticker

                if quandl_failure_count < 10:
                    try:
                        data, _ = get_q_data(ticker, start_date=date1, end_date=date2, adjusted=True, output_format='df')
                    except:
                        quandl_failure_count += 1
                        data, _ = get_ts_data(ticker, interval="D", outputsize=output_size, adjusted=True, output_format='df')
                        #number_days = (date2 - date1).days - 1
                        number_days = (date2 - date1).days
                        data = data[-number_days:]
                    '''
                    # start of trial code
                    data, _ = get_ts_data(ticker, interval="D", outputsize=output_size, adjusted=True, output_format='df')
                    #number_days = (date2 - date1).days - 1
                    number_days = (date2 - date1).days
                    data = data[-number_days:]
                    # end of trial code
                    '''
                else:
                    quandl_failure_count += 1
                    data, _ = get_ts_data(ticker, interval="D", outputsize=output_size, adjusted=True, output_format='df')
                    #number_days = (date2 - date1).days - 1
                    number_days = (date2 - date1).days
                    data = data[-number_days:]


                data = data[items]
                #print " ... got data[items] ..."
                number_tries = 0
                #print " last 5 yahoo values = ", data.values[-5:]
            except:
                pass
            """

            ###data = get_data_yahoo(ticker, start = date1, end = date2)[items]
            ####data = get_data_yahoo_fix(ticker, start = date1, end = date2)[items]
            if (date2 - date1).days <= 100:
                output_size = 'compact'
            else:
                output_size = 'full'
            #print "   ... ticker = ", ticker

            for ii in range(25):
                try:
                    dataseries, _ = get_ts_data(ticker, interval="D", outputsize=output_size, adjusted=True, output_format='df')
                    data = pd.DataFrame({'dates':dataseries.index, 'Adj Close':dataseries['Adj Close'].values})
                    data.set_index('dates', inplace=True)
                    number_days = (date2 - date1).days
                    data = data[-number_days:]
                    break
                except:
                    from time import sleep
                    sleep(3)
                    pass

        else:
            #print "number_tries = ", number_tries, " trying with google"
            print("   ...retrieving quotes using google")
            try:
                data = get_data_google(ticker, start = date1, end = date2)[google_items]
                number_tries = 0
            except:
                pass
        #print ' data = ', data
        dates = data.index
        #print " ... got dates from df.index ..."
        #print ' dates = ', dates
        dates = [d.to_pydatetime() for d in dates]
        #print " ... dates reformatted ..."

        data.columns = [ticker]
        #print " ... dates column name changed ..."
        #print ' ticker = ', [ticker]
        #print ' data.columns = ', data.columns
        if Verbose:
            print(i," of ",len(tickers)," ticker ",ticker," has ",data.shape[0]," quotes")

        if itick-re_tries == 0:
            #print " creating dataframe..."
            quotes_df = data
        else:
            #print " joining to dataframe..."
            quotes_df = quotes_df.join( data, how='outer' )
            #print " joined to dataframe..."
        i += 1

    print("number of tickers successfully processed = ", i)
    if i > 0 :
        quotes_df.sort_index( axis=0, inplace=True )
        return quotes_df

    else :
        # return empty DataFrame
        quotes_df = pd.DataFrame( [0,0], ['Dates',date2])
        quotes_df.columns = ['None']

    return quotes_df


def get_pe(ticker):

    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    import numpy as np

    try:
        url2 = 'http://finance.google.com/finance?q=NASDAQ%3A'+ticker
        text_soup = BeautifulSoup(urlopen(url2).read(), 'lxml') #read in

        pe = np.nan
        titles = text_soup.findAll('td', {'class': 'key'})
        for title in titles:
            if 'P/E' in title.text:
                pe_text =  [td.text for td in title.findNextSiblings(attrs={'class': 'val'}) if td.text]
                pe = float(pe_text[0].split("\n")[0])
    except:
        pe = np.nan

    print(" switch to get_pe in quotes_for_list_adjClose.py")
    return pe


def get_pe(ticker):

    from functions.quotes_for_list_adjClose import get_pe_finviz
    return get_pe_finviz(ticker, verbose=True)


def get_quote_and_pe(ticker):

    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    import numpy as np

    url2 = 'http://finance.google.com/finance?q=NASDAQ%3A'+ticker
    text_soup = BeautifulSoup(urlopen(url2).read(), 'lxml') #read in

    pe = np.nan
    titles = text_soup.findAll('td', {'class': 'key'})
    for title in titles:
        if 'P/E' in title.text:
            pe_text =  [td.text for td in title.findNextSiblings(attrs={'class': 'val'}) if td.text]
            pe = float(pe_text[0].split("\n")[0])

    try:
        titles = text_soup.findAll('meta', {'itemprop': 'price'})
        for title in titles:
            price = float(title['content'])
    except:
        price = np.nan

    return price, pe
