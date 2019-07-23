'''
Created on May 12, 202

@author: donaldpg
'''

def get_RU1000List( verbose=True ):
    # set some excluded tickers that are note downloaded by yah**
    excluded_tickers = ['ACET', 'ACXM', 'BOFI', 'GOV', 'HBHC', 'HYH'] # 'OCLR', 'SONC'
    
    ###
    ### Query wikipedia.com for updated list of stocks in S&P 1000 index.
    ### (same as Russell 1000?)
    ### Return list with stock tickers.
    ###
    import urllib
    import os
    import datetime
    from bs4 import BeautifulSoup

    ###
    ### get symbol list from previous period
    ###
    symbol_directory = os.path.join( os.getcwd(), "symbols" )

    symbol_file = "RU1000_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    with open(symbols_file, "r+") as f:
        old_symbolList = f.readlines()
    for i in range( len(old_symbolList) ) :
        old_symbolList[i] = old_symbolList[i].replace("\n","")

    ###
    ### get current symbol list from wikipedia website
    ###

    base_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_1000_companies'
    #content = urllib2.urlopen(base_url)
    #c = content.read()
    #print "\n\n\n content = ", content
    #print "... got web content"

    soup = BeautifulSoup( urllib.request.urlopen(base_url), "lxml" )
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
            '''
            #_ticker = col[0].string.strip()
            _ticker = str(col[0]).split('"')[-1].split(">")[1].split("<")[0]
            _ticker = _ticker.replace(".","-")
            try:
                _company = col[1].string.strip()
            except:
                _company = str(col[1]).split('title="')[-1].split('">')[0]
            _sector = col[3].string.strip()
            _subIndustry = col[4].string.strip()
            symbolList.append(_ticker)
            companyNamesList.append(_company)
            industry.append(_sector)
            subIndustry.append(_subIndustry)
            '''
            cols = [ele.text.strip() for ele in col]
            symbolList.append(cols[1])
            companyNamesList.append(cols[0])
            industry.append(cols[2])
            subIndustry.append(cols[3])
        except:
            pass
    print("... retrieved RU1000 companies lists from internet")
    
    # remove excluded_tickers
    subIndustry = [subIndustry[i] for i in range(len(symbolList)) if symbolList[i] not in excluded_tickers]
    industry = [industry[i] for i in range(len(symbolList)) if symbolList[i] not in excluded_tickers]
    companyNamesList = [companyNamesList[i] for i in range(len(symbolList)) if symbolList[i] not in excluded_tickers]
    symbolList = [symbolList[i] for i in range(len(symbolList)) if symbolList[i] not in excluded_tickers]

    companyName_file = os.path.join( symbol_directory, "RU1000_companyNames.txt" )
    with open( companyName_file, "w" ) as f:
        for i in range( len(symbolList) ) :
            f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )
    print("... wrote 1000_companyNames.txt")

    ###
    ### compare old list with new list and print changes, if any
    ###

    # file for index changes history
    symbol_change_file = "RU1000_symbolsChanges.txt"
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
                print((" Ticker ", ticker, " has been removed from the RU1000 index"))
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
                print((" Ticker ", ticker, " has been added to the RU1000 index"))
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
        symbol_file = "RU1000_Symbols.txt"
        archive_symbol_file = "RU1000_symbols__" + str(datetime.date.today()) + ".txt"
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



def get_SP500List( verbose=True ):
    ###
    ### Query wikipedia.com for updated list of stocks in S&P 500 index.
    ### Return list with stock tickers.
    ###
    import urllib
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

    soup = BeautifulSoup( urllib.request.urlopen(base_url), "lxml" )
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
            _ticker = col[0].text.strip()
            _ticker = _ticker.replace(".","-")
            try:
                _company = col[1].text.strip()
            except:
                _company = str(col[1]).split('title="')[-1].split('">')[0]
            _sector = col[3].text.strip()
            _subIndustry = col[4].text.strip()
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
        #
        # Note: Could use the following url to download a csv file containing
        #       a list with columns for ticker, company name, sector, and industry.
        #       http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&render=download
        #       - it includes about 6700 companies as of 2017-1-8
        #       - does not include ETF or mutual funds
        #
        ##
        ## The method used is specific to the nasdaq-100 index
        ##
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
        try:
           with open(symbols_changes_file, 'x') as f:
               pass
        except FileExistsError:
           pass
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
        removedTickersText = ''
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



def readSymbolList(filename,verbose=False):
    import os
    # Get the Data
    try:
        print(" ...inside readSymbolList... filename = ", filename)
        print(" ... current directory: " + os.getcwd())
        infile = open(filename,"r")
    except:
        symbol_directory = os.path.join( os.getcwd(), "symbols" )
        # the symbols list doesn't exist. generate from the web.
        if 'SP500' in filename:
            symbol_file = "SP500_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_SP500List( verbose=True )
            try:
                infile.close()
            except:
                pass
            infile = open(filename,"r")
        elif 'Naz100' in filename:
            symbol_file = "Naz100_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_Naz100List( verbose=True )
        elif 'RU1000' in filename or 'SP1000' in filename:
            symbol_file = "RU1000_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_RU1000List( verbose=True )


    symbols = []

    content = infile.read()
    number_lines = len(content.split("\n"))
    if number_lines == 0:
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
        elif 'RU1000' in filename or 'SP1000' in filename:
            symbol_file = "RU1000_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_RU1000List( verbose=True )

    infile.close()
    infile = open(filename,"r")

    while infile:
        line = infile.readline()
        s = line.split()
        n = len(s)
        if n != 0:
            for i in range(len(s)):
                s[i] = s[i].replace('.','-')
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
