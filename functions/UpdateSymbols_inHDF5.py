
import datetime
from datetime import timedelta
import pandas as pd
import nose
import os

from matplotlib.pylab import *

from scipy.stats import gmean

## local imports
from functions.quotes_for_list_adjClose import *
from functions.TAfunctions import *
from functions.readSymbols import *

def loadQuotes_fromHDF( symbols_file ):

    (directory_name, file_name) = os.path.split(symbols_file)
    (shortname, extension) = os.path.splitext( file_name )

    print(("file name for symbols = ","_"+shortname+"_"))
    print(("file type for symbols = ",extension))

    # set up to write quotes to disk.

    if shortname == "symbols" :
        listname = "TAA-Symbols"
    elif shortname == "cmg_symbols" :
        listname = "CMG-Symbols"
    elif shortname == "Naz100_Symbols" :
        listname = "Naz100_Symbols"
    elif shortname == "biglist" :
        listname = "biglist-Symbols"
    elif shortname == "ETF_symbols" :
        listname = "ETF-Symbols"
    elif shortname == "ProvidentFundSymbols" :
        listname = "ProvidentFund-Symbols"
    elif shortname == "SP500_Symbols" :
        listname = "SP500-Symbols"
    else :
        listname = shortname

    hdf5_directory = os.path.join( os.getcwd(), "symbols" )
    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")

    print("")
    print("")
    print(("symbol_directory = ", directory_name))
    print(("symbols_file = ", symbols_file))
    print(("shortname, extension = ",shortname, extension))
    print(("hdf5filename = ",hdf5filename))


    try:
        quote = pd.read_hdf( hdf5filename, listname )
        x = quote.as_matrix()
        x = x.swapaxes(0,1)
        date = quote.index
        symbols = list(quote.columns.values)
        dates=[]
        for i in range(len(date)):
            datestr = date[i]
            date_newformat = datetime.date(*[int(val) for val in datestr.split('-')])
            dates.append(date_newformat)
    except:
        createHDF( hdf5_directory, symbols_file, listname )
        quote = pd.read_hdf( hdf5filename, listname )
        x = quote.as_matrix()
        x = x.swapaxes(0,1)
        date = quote.index
        symbols = list(quote.columns.values)

        dates=[]
        for i in range(len(date)):
            datestr = date[i]
            date_newformat = datetime.date(*[int(val) for val in datestr.split('-')])
            date_newformat = datestr
            dates.append(date_newformat)

    return x, symbols, dates, quote, listname

def cleanup_quotes(symbols_file,newquotesfirstdate, newquoteslastdate):
    # compare quotes currently on hdf with updated quotes from internet.
    print(" ...   inside compareHDF_and_newquotes   ...")
    print((" ... newquotesfirstdate = ", newquotesfirstdate))
    print((" ... newquoteslastdate = ", newquoteslastdate))

    # get existing quotes from hdf
    (directory_name, file_name) = os.path.split(symbols_file)
    (shortname, extension) = os.path.splitext( file_name )

    print(("file name for symbols = ","_"+shortname+"_"))
    print(("file type for symbols = ",extension))

    # set up to write quotes to disk.

    if shortname == "symbols" :
        listname = "TAA-Symbols"
    elif shortname == "cmg_symbols" :
        listname = "CMG-Symbols"
    elif shortname == "Naz100_Symbols" :
        listname = "Naz100_Symbols"
    elif shortname == "biglist" :
        listname = "biglist-Symbols"
    elif shortname == "ETF_symbols" :
        listname = "ETF-Symbols"
    elif shortname == "ProvidentFundSymbols" :
        listname = "ProvidentFund-Symbols"
    elif shortname == "sp500_symbols" :
        listname = "SP500-Symbols"
    else :
        listname = shortname

    hdf5_directory = os.path.join( os.getcwd(), "symbols" )
    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")

    print("")
    print("")
    print(("symbol_directory = ", directory_name))
    print(("symbols_file = ", symbols_file))
    print(("shortname, extension = ",shortname, extension))
    print(("hdf5filename = ",hdf5filename))

    dataframeFromHDF = pd.read_hdf( hdf5filename, listname )
    x_hdf = dataframeFromHDF.as_matrix()
    x_hdf = x_hdf.swapaxes(0,1)
    date_hdf = dataframeFromHDF.index
    symbols_hdf = list(dataframeFromHDF.columns.values)

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    #for ii in range(x.shape[0]):
    for ii,isymbolupdate in enumerate(symbols_hdf):
        xupdate = dataframeFromHDF[isymbolupdate]
        '''
        if isymbolupdate == 'SBUX':
            import pdb
            pdb.set_trace()
        '''
        print((" ... cleanup_quotes ... symbol = ", isymbolupdate))
        xupdate = cleanspikes(xupdate)
        xupdate = interpolate(xupdate)
        xupdate = cleantobeginning(xupdate)
        dataframeFromHDF[isymbolupdate] = xupdate.copy()
        #xupdate[ii,:] = np.array(xupdate[ii,:]).astype('float')
        #xupdate[ii,:] = interpolate(xupdate[ii,:])
        #xupdate[ii,:] = cleantobeginning(xupdate[ii,:])

    dataframeFromHDF.to_hdf( hdf5filename, listname, mode='a',format='table',append=False,complevel=5,complib='blosc')

    return


def compareHDF_and_newquotes(symbols_file,newquotesfirstdate, newquoteslastdate):
    # compare quotes currently on hdf with updated quotes from internet.
    print(" ...   inside compareHDF_and_newquotes   ...")
    print((" ... newquotesfirstdate = ", newquotesfirstdate))
    print((" ... newquoteslastdate = ", newquoteslastdate))

    # get existing quotes from hdf
    (directory_name, file_name) = os.path.split(symbols_file)
    (shortname, extension) = os.path.splitext( file_name )

    print(("file name for symbols = ","_"+shortname+"_"))
    print(("file type for symbols = ",extension))

    # set up to write quotes to disk.

    if shortname == "symbols" :
        listname = "TAA-Symbols"
    elif shortname == "cmg_symbols" :
        listname = "CMG-Symbols"
    elif shortname == "Naz100_Symbols" :
        listname = "Naz100_Symbols"
    elif shortname == "biglist" :
        listname = "biglist-Symbols"
    elif shortname == "ETF_symbols" :
        listname = "ETF-Symbols"
    elif shortname == "ProvidentFundSymbols" :
        listname = "ProvidentFund-Symbols"
    elif shortname == "sp500_symbols" :
        listname = "SP500-Symbols"
    else :
        listname = shortname

    hdf5_directory = os.path.join( os.getcwd(), "symbols" )
    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")

    print("")
    print("")
    print(("symbol_directory = ", directory_name))
    print(("symbols_file = ", symbols_file))
    print(("shortname, extension = ",shortname, extension))
    print(("hdf5filename = ",hdf5filename))

    dataframeFromHDF = pd.read_hdf( hdf5filename, listname )
    x_hdf = dataframeFromHDF.as_matrix()
    x_hdf = x_hdf.swapaxes(0,1)
    date_hdf = dataframeFromHDF.index
    symbols_hdf = list(dataframeFromHDF.columns.values)

    # get new quotes dataframe from internet
    newadjClose, newsymbols, newdatearray = arrayFromQuotesForList(symbols_file, newquotesfirstdate, newquoteslastdate)
    print((" security values check: ",newadjClose[isnan(newadjClose)].shape))
    newdates = []
    for i in range(newdatearray.shape[0]):
        newdates.append(str(newdatearray[i]))
    #quotes_NewSymbols = pd.DataFrame(newadjClose, [symbols,newdates], dtype=float)
    dataframeFromInternet = pd.DataFrame(newadjClose.swapaxes(0,1), index=newdates, columns=newsymbols)

    ###################
    from functions.TAfunctions import interpolate
    from functions.TAfunctions import cleantobeginning

    # clean up quotes for missing values and varying starting date
    #x = quote.as_matrix().swapaxes(0,1)
    ##xupdate = dataframeFromInternet.values.T
    symbolListupdate = list(dataframeFromInternet.columns.values)

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    #for ii in range(x.shape[0]):
    for ii,isymbolupdate in enumerate(symbolListupdate):
        xupdate = dataframeFromInternet[isymbolupdate]
        '''
        if isymbolupdate == 'SBUX':
            import pdb
            pdb.set_trace()
        '''
        print((" isymbolupdate,xupdate = ", isymbolupdate,xupdate.as_matrix()))
        xupdate = cleanspikes(xupdate)
        xupdate = interpolate(xupdate)
        xupdate = cleantobeginning(xupdate)
        #xupdate[ii,:] = np.array(xupdate[ii,:]).astype('float')
        #xupdate[ii,:] = interpolate(xupdate[ii,:])
        #xupdate[ii,:] = cleantobeginning(xupdate[ii,:])
    ###################

    x_net = dataframeFromInternet.as_matrix()
    x_net = x_net.swapaxes(0,1)
    date_net = dataframeFromInternet.index
    symbols_net = list(dataframeFromInternet.columns.values)

    # find joined symbols
    symbols_all = symbols_hdf + symbols_net
    symbols_all = list(set(symbols_all))
    symbols_all.sort()

    for isymbol in symbols_all:
        # find date range for shorter of quotes update from net or on hdf
        try:
            hdf_index = symbols_hdf.index(isymbol)
            firstindexup_hdf = np.argmax(np.clip(x_hdf[hdf_index,:]/x_hdf[hdf_index,0],1.,1.+1.e-5))
            firstindexdown_hdf = np.argmin(np.clip(x_hdf[hdf_index,:]/x_hdf[hdf_index,0],1.-1.e-5,1.))
            firstindex_hdf = max(firstindexup_hdf,firstindexdown_hdf)

            net_index = symbols_net.index(isymbol)
            firstindexup_net = np.argmax(np.clip(x_net[net_index,:]/x_net[net_index,0],1.,1.+1.e-5))
            firstindexdown_net = np.argmin(np.clip(x_net[net_index,:]/x_net[net_index,0],1.-1.e-5,1.))
            firstindex_net = max(firstindexup_net,firstindexdown_net)
            firstDate = max( date_net[firstindex_net], date_hdf[firstindex_hdf] )
            lastDate = min( date_net[-1], date_hdf[-1] )

            values_hdf = x_hdf[hdf_index,list(date_hdf).index(firstDate):list(date_hdf).index(lastDate)+1]
            values_net = x_net[net_index,list(date_net).index(firstDate):list(date_net).index(lastDate)+1]

            if False in values_hdf==values_net:
                print((" ... **** symbol ", format(isymbol,'5s'), " is different in hdf and update from internet (", firstDate, " to ", lastDate, " )"))
            else:
                print((" ... symbol ", format(isymbol,'5s'), " is same in hdf and update from internet (", firstDate, " to ", lastDate, " )"))
        except:
            print((" ... **** **** symbol ", format(isymbol,'5s'), " not matched in hdf and update from internet"))

        '''
        if isymbol == 'SBUX':
            print " .... firstdate, lastdate = ", firstDate, lastDate
            datesForPlot = date_hdf[list(date_hdf).index(firstDate):list(date_hdf).index(lastDate)+1]
            _datesForPlot1=[]
            for i in range(len(datesForPlot)):
                datestr = datesForPlot[i]
                date_newformat = datetime.date(*[int(val) for val in datestr.split('-')])
                #date_newformat = datestr
                _datesForPlot1.append(date_newformat)
                iindex = list(date_hdf).index(firstDate) + i
                #print "i,date_newformat,values_net['SBUX'] = ", i,date_newformat,x_net[net_index,iindex]

            print " .... _datesForPlot1 = ", _datesForPlot1
            plt.figure()
            plt.grid()
            plt.plot(_datesForPlot1,values_hdf)
            plt.plot(_datesForPlot1,values_hdf,'b.')

            datesForPlot = date_net[list(date_net).index(firstDate):list(date_net).index(lastDate)+1]
            _datesForPlot2=[]
            for i in range(len(datesForPlot)):
                datestr = datesForPlot[i]
                date_newformat = datetime.date(*[int(val) for val in datestr.split('-')])
                #date_newformat = datestr
                _datesForPlot2.append(date_newformat)
            print "\n\n\n .... _datesForPlot2 = ", _datesForPlot2
            plt.plot(_datesForPlot2,values_net)
            plt.plot(_datesForPlot2,values_net,'g.')
        '''

    return



def getLastDateFromHDF5( symbol_directory, symbols_file ) :
    filename = os.path.join(symbol_directory, symbols_file)
    adjClose, symbols, datearray, quote, _ = loadQuotes_fromHDF( filename )
    import numpy as np
    symbols2 = list(data.columns.values)
    for i in range(len(symbols)):
        numisnans = adjClose[i,:].copy()
    numdates = adjClose.shape[1]
    for i in range(adjClose.shape[0]):
        for j in range(adjClose.shape[1]):
            if isnan(adjClose[i,numdates-j-1])  :
                lastindex = numdates-j-1
    return datearray[lastindex]


def getLastDateFromHDF5( symbol_directory, symbols_file ) :
    filename = os.path.join(symbol_directory, symbols_file)
    _, _, datearray, _, _ = loadQuotes_fromHDF( filename )

    import datetime
    today = datetime.datetime.now()
    hourOfDay = today.hour
    dayOfWeek = today.weekday()
    dayOfMonth = today.day
    tomorrow = today + datetime.timedelta( days=1 )
    yesterday = datearray[-1] - datetime.timedelta( days=1 )
    tomorrowDayOfMonth = tomorrow.day

    # set up to return current day's quotes.
    # - Except late Friday nights and at end of month, when quotes are updated for entire history.
    # - This logic ensures dividends and splits are accounted for.
    # TODO: check if there's a split or deividend and only get entire history if 'yes'.

    if  hourOfDay >= 22 :
        return datearray[0]
    else:
        return yesterday

"""
def UpdateHDF5( symbol_directory, symbols_file ):

    ##
    ##  Update symbols in 'symbols_file' with quotes more recent than last update.
    ##

    filename = os.path.join(symbol_directory, symbols_file)

    x, symbols, datearray, quote, listname = loadQuotes_fromHDF( filename )

    # get last date in hdf5 archive
    #from datetime import datetime
    import datetime

    date = quote.index
    lastdate = getLastDateFromHDF5( symbol_directory, symbols_file )

    ##
    ## Get quotes for each symbol in list
    ## process dates.
    ## Clean up quotes.
    ## Make a plot showing all symbols in list
    ##

    # locate symbols added to list that aren't in HDF5 file
    symbols_in_list = readSymbolList( filename, verbose=False)
    symbols_in_HDF5 = list(quote.columns.values)
    new_symbols = [x for x in symbols_in_list if x  not in symbols_in_HDF5]

    # write new symbols to temporary file
    if len(new_symbols) > 0:
        # write new symbols to temporary file
        tempfilename = os.path.join(symbol_directory, "newsymbols_tempfile.txt")
        OUTFILE = open(tempfilename,"w",0)
        for i,isymbol in enumerate(new_symbols):
            print "new symbol = ", isymbol
            OUTFILE.write(str(isymbol) + "\n")

        newquotesfirstdate = datetime.date(1991,1,1)
        newquoteslastdate = datetime.date.today()

        # print dates to be used
        print "dates for new symbol found = ", newquotesfirstdate, newquoteslastdate

        newadjClose, newsymbols, newdatearray = arrayFromQuotesForList(tempfilename, newquotesfirstdate, newquoteslastdate)

        print " security values check: ",newadjClose[isnan(newadjClose)].shape

        newdates = []
        for i in range(newdatearray.shape[0]):
            newdates.append(str(newdatearray[i]))
        #quotes_NewSymbols = pd.DataFrame(newadjClose, [symbols,newdates], dtype=float)
        quotes_NewSymbols = pd.DataFrame(newadjClose.swapaxes(0,1), index=newdates, columns=newsymbols)

    '''
    # add CASH as symbol (with value of 1.0 for all dates)
    CASHsymbols = ['CASH']
    if len(CASHsymbols) > 0:
        # write CASH symbols to temporary file
        tempfilename = os.path.join(symbol_directory, "CASHsymbols_tempfile.txt")
        OUTFILE = open(tempfilename,"w",0)
        for i,isymbol in enumerate(CASHsymbols):
            print "new symbol = ", isymbol
            OUTFILE.write(str(isymbol) + "\n")

        date = quote.index
        CASHdatearray = np.array( date )
        CASHadjClose = np.ones( (1,len(CASHdatearray)), float ) * 100.
        for i in range(CASHadjClose.shape[0]):
            if i%10 == 0:
                CASHadjClose[i] = CASHadjClose[i-1] + .01
            else:
                CASHadjClose[i] = CASHadjClose[i-1]

        print " security values check: ",CASHadjClose[isnan(CASHadjClose)].shape
        print "CASHsymbols = ", CASHsymbols

        CASHdates = []
        for i in range(CASHdatearray.shape[0]):
            CASHdates.append(str(CASHdatearray[i]))
        print "sizes of CASHadjClose, CASHdates = ",CASHadjClose.shape, len(CASHdates)
        #quotes_CASHSymbols = la.larry(CASHadjClose, [CASHsymbols,CASHdates], dtype=float)
        quotes_CASHSymbols = pd.DataFrame( CASHadjClose.swapaxes(0,1), index=CASHdates, columns=CASHsymbols)
        print "\n ... inside UpdateHDF5 ... finished creating quotes_CASHSymbols ..."
    '''

    ##
    ## Get quotes for each symbol in list
    ## process dates.
    ## Clean up quotes.
    ## Make a plot showing all symbols in list
    ##

    if type(lastdate) == str:
        newquotesfirstdate = datetime.date(*[int(val) for val in lastdate.split('-')])
    else:
        newquotesfirstdate = lastdate
    today = datetime.datetime.now()
    tomorrow = today + timedelta( days=1 )
    newquoteslastdate = tomorrow

    '''
    ### start temporary test
    f = open(filename,'r')
    symbols_from_file = f.read()
    print " ... inside UpdateHDF... filename = ", filename
    print " ... inside UpdateHDF... symbols_from_file = ", symbols_from_file
    ### end temporary test
    '''

    newadjClose, symbols, newdatearray = arrayFromQuotesForList(filename, newquotesfirstdate, newquoteslastdate)

    print " security values check: ",newadjClose[isnan(newadjClose)].shape

    newdates = []
    for i in range(newdatearray.shape[0]):
        newdates.append(str(newdatearray[i]))
    #quoteupdate = la.larry(newadjClose, [symbols,newdates], dtype=float)
    quoteupdate = pd.DataFrame( newadjClose.swapaxes(0,1), index=newdates, columns=symbols)


    #updatedquotes = quote.combine_first( quoteupdate )
    updatedquotes = quoteupdate.combine_first( quote )

    ###################
    from functions.TAfunctions import cleanspikes
    from functions.TAfunctions import interpolate
    from functions.TAfunctions import cleantobeginning

    # clean up quotes for missing values and varying starting date
    #x = quote.as_matrix().swapaxes(0,1)
    xupdate = quoteupdate.values.T
    symbolListupdate = list(quoteupdate.columns.values)

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    #for ii in range(x.shape[0]):
    for ii,isymbolupdate in enumerate(symbolListupdate):
        '''
        if ii%5 == 0:
            print "  ... progress:  ii, symbol = ", ii, isymbolupdate
        '''
        xupdate = quoteupdate[isymbolupdate]
        xupdate = cleanspikes(xupdate)
        xupdate = interpolate(xupdate)
        xupdate = cleantobeginning(xupdate)
        #xupdate[ii,:] = np.array(xupdate[ii,:]).astype('float')
        #xupdate[ii,:] = interpolate(xupdate[ii,:])
        #xupdate[ii,:] = cleantobeginning(xupdate[ii,:])
    ###################

    if len(new_symbols) > 0:
        print "\n\n\n...quotes_NewSymbols = ", quotes_NewSymbols.info()
        print "\n\n\n...updatedquotes = ", updatedquotes.info()
        updatedquotes = updatedquotes.join( quotes_NewSymbols, how='outer' )
        ###updatedquotes = pd.merge( updatedquotes,quotes_NewSymbols, 'outer' )
        print "\n\n\n...merged updatedquotes = ", updatedquotes.info()

    ###if len(CASHsymbols) > 0:
    CASHsymbols = ['CASH']
    if len(CASHsymbols) > 0:
        # write CASH symbols to temporary file
        tempfilename = os.path.join(symbol_directory, "CASHsymbols_tempfile.txt")
        OUTFILE = open(tempfilename,"w",0)
        for i,isymbol in enumerate(CASHsymbols):
            print "new symbol = ", isymbol
            OUTFILE.write(str(isymbol) + "\n")

        date = updatedquotes.index
        CASHdatearray = np.array( date )
        ##CASHadjClose = np.ones( (1,len(CASHdatearray)), float ) * 100.
        CASHadjClose = np.ones( (1,len(date)), float ) * 100.
        for i in range(CASHadjClose.shape[0]):
            if i%10 == 0:
                CASHadjClose[i] = CASHadjClose[i-1] + .01
            else:
                CASHadjClose[i] = CASHadjClose[i-1]

        print " security values check: ",CASHadjClose[isnan(CASHadjClose)].shape
        print "CASHsymbols = ", CASHsymbols

        CASHdates = []
        for i in range(CASHdatearray.shape[0]):
            CASHdates.append(str(CASHdatearray[i]))
        print "sizes of CASHadjClose, CASHdates = ",CASHadjClose.shape, len(CASHdates)
        #quotes_CASHSymbols = la.larry(CASHadjClose, [CASHsymbols,CASHdates], dtype=float)
        ##quotes_CASHSymbols = pd.DataFrame( CASHadjClose.swapaxes(0,1), index=CASHdates, columns=CASHsymbols)
        quotes_CASHSymbols = pd.DataFrame( CASHadjClose.swapaxes(0,1), index=date, columns=CASHsymbols)
        print "\n ... inside UpdateHDF5 ... finished creating quotes_CASHSymbols ..."

        print "\n\n\n...quotes_CASHSymbols = ", quotes_CASHSymbols.info()
        try:
            del updatedquotes['CASH']
        except:
            pass
        ##updatedquotes = updatedquotes.join( quotes_CASHSymbols, how='outer' )
        updatedquotes = pd.merge( updatedquotes, quotes_CASHSymbols, 'outer' )
        '''
        try:
            updatedquotes['CASH'] = np.ones( len(updatedquotes['CASH']), 'float' )
        except:
            print " ...inside UpdateHDF5... Error setting updatedquotes['CASH'] to np.ones "
        '''

    # set up to write quotes to disk.
    dirname = os.path.join( os.getcwd(), "symbols" )

    hdf5filename = os.path.join( dirname, listname + "_.hdf5" )
    print "hdf5 filename = ",hdf5filename
    updatedquotes.to_hdf( hdf5filename, listname, mode='a',format='table',append=False,complevel=5,complib='blosc')

    ###cleanup_quotes(filename, datearray[0], today)

    return
"""

def UpdateHDF5( symbol_directory, symbols_file ):

    ##
    ##  Update symbols in 'symbols_file' with quotes more recent than last update.
    ##

    filename = os.path.join(symbol_directory, symbols_file)

    x, symbols, datearray, quote, listname = loadQuotes_fromHDF( filename )

    # get last date in hdf5 archive
    #from datetime import datetime
    import datetime

    date = quote.index
    lastdate = getLastDateFromHDF5( symbol_directory, symbols_file )

    ##
    ## Get quotes for each symbol in list
    ## process dates.
    ## Clean up quotes.
    ## Make a plot showing all symbols in list
    ##

    # locate symbols added to list that aren't in HDF5 file
    symbols_in_list = readSymbolList( filename, verbose=False)
    symbols_in_HDF5 = list(quote.columns.values)
    new_symbols = [x for x in symbols_in_list if x  not in symbols_in_HDF5]

    # write new symbols to temporary file
    if len(new_symbols) > 0:
        # write new symbols to temporary file
        tempfilename = os.path.join(symbol_directory, "newsymbols_tempfile.txt")
        OUTFILE = open(tempfilename,"w",0)
        for i,isymbol in enumerate(new_symbols):
            print(("new symbol = ", isymbol))
            OUTFILE.write(str(isymbol) + "\n")

        newquotesfirstdate = datetime.date(1991,1,1)
        newquoteslastdate = datetime.date.today()

        # print dates to be used
        print(("dates for new symbol found = ", newquotesfirstdate, newquoteslastdate))

        newadjClose, newsymbols, newdatearray = arrayFromQuotesForList(tempfilename, newquotesfirstdate, newquoteslastdate)

        print((" security values check: ",newadjClose[isnan(newadjClose)].shape))

        newdates = []
        for i in range(newdatearray.shape[0]):
            newdates.append(str(newdatearray[i]))
        #quotes_NewSymbols = pd.DataFrame(newadjClose, [symbols,newdates], dtype=float)
        quotes_NewSymbols = pd.DataFrame(newadjClose.swapaxes(0,1), index=newdates, columns=newsymbols)

    ##
    ## Get quotes for each symbol in list
    ## process dates.
    ## Clean up quotes.
    ## Make a plot showing all symbols in list
    ##

    if type(lastdate) == str:
        newquotesfirstdate = datetime.date(*[int(val) for val in lastdate.split('-')])
    else:
        newquotesfirstdate = lastdate
    today = datetime.datetime.now()
    tomorrow = today + timedelta( days=1 )
    newquoteslastdate = tomorrow


    newadjClose, symbols, newdatearray = arrayFromQuotesForList(filename, newquotesfirstdate, newquoteslastdate)

    print((" security values check: ",newadjClose[isnan(newadjClose)].shape))

    newdates = []
    for i in range(newdatearray.shape[0]):
        newdates.append(str(newdatearray[i]))
    quoteupdate = pd.DataFrame( newadjClose.swapaxes(0,1), index=newdates, columns=symbols)

    updatedquotes = quoteupdate.combine_first( quote )

    ###################
    from functions.TAfunctions import cleanspikes
    from functions.TAfunctions import interpolate
    from functions.TAfunctions import cleantobeginning

    # clean up quotes for missing values and varying starting date
    #x = quote.as_matrix().swapaxes(0,1)
    xupdate = updatedquotes.values.T
    symbolListupdate = list(updatedquotes.columns.values)

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    #for ii in range(x.shape[0]):
    for ii,isymbolupdate in enumerate(symbolListupdate):
        '''
        if ii%5 == 0:
            print "  ... progress:  ii, symbol = ", ii, isymbolupdate
        '''
        xupdate = updatedquotes[isymbolupdate].values
        xupdate = cleanspikes(xupdate)
        xupdate = interpolate(xupdate)
        xupdate = cleantobeginning(xupdate)
        updatedquotes[isymbolupdate] = xupdate
    ###################

    if len(new_symbols) > 0:
        print(("\n\n\n...quotes_NewSymbols = ", quotes_NewSymbols.info()))
        print(("\n\n\n...updatedquotes = ", updatedquotes.info()))
        for isymbol in new_symbols:
            updatedquotes[isymbol] = quotes_NewSymbols[isymbol]
        print(("\n\n\n...merged updatedquotes = ", updatedquotes.info()))


    CASHadjClose = np.ones( (len(updatedquotes.index)), float ) * 100.
    for i in range(CASHadjClose.shape[0]):
        if i%10 == 0:
            CASHadjClose[i] = CASHadjClose[i-1] + .01
        else:
            CASHadjClose[i] = CASHadjClose[i-1]

    updatedquotes['CASH'] = CASHadjClose


    # set up to write quotes to disk.
    dirname = os.path.join( os.getcwd(), "symbols" )

    hdf5filename = os.path.join( dirname, listname + "_.hdf5" )
    print(("hdf5 filename = ",hdf5filename))
    updatedquotes.to_hdf( hdf5filename, listname, mode='a',format='table',append=False,complevel=5,complib='blosc')

    return




def createHDF( hdf5_directory, symbol_file, listname ):

    import os
    import numpy as np
    
    import matplotlib.gridspec as gridspec
    import pandas as pd

    from datetime import datetime

    import nose

    ## local imports
    
    

    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")
    (shortname, extension) = os.path.splitext( symbol_file )
    symbol_directory = hdf5_directory

    print(("symbol_directory = ",symbol_directory))
    print(("symbol_file = ",symbol_file))
    print(("shortname, extension = ",shortname, extension))
    print(("hdf5filename = ",hdf5filename))

    ##
    ## Get quotes for each symbol in list
    ## process dates.
    ## Clean up quotes.
    ## Make a plot showing all symbols in list
    ##

    import datetime
    firstdate=datetime.date(1991,1,1)
    lastdate=(2011,11,30)
    lastdate=(2013,6,1)
    today = datetime.date.today()
    lastdate = today

    filename = os.path.join( symbol_directory, symbol_file )
    print(("filename with list of symbols = ", filename))

    adjClose, symbols, datearray = arrayFromQuotesForList(filename, firstdate, lastdate)

    print((" security values check (adjClose): ",adjClose[isnan(adjClose)].shape))

    dates = datearray
    quotes_df = pd.DataFrame( adjClose.swapaxes(0,1), index=datearray, columns=symbols)

    # write pandas dataframe to hdf
    quotes_df.to_hdf( hdf5filename, listname, mode='a',format='table',append=False,complevel=5,complib='blosc')

    return
