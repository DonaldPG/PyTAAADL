import os
import configparser
import numpy as np

def from_config_file(config_filename):
    with open(config_filename, "r") as fid:
        config = configparser.ConfigParser()
        params = config.readfp(fid)
    return params

def GetParams():
    ######################
    ### Input parameters
    ######################

    # set default values
    defaults = { "Runtime": ["2 days"], "Pausetime": ["1 hours"] }
    params = {}

    # read the parameters form the configuration file
    config_filename = "PyTAAADL.params"

    #params = from_config_file(config_filename)
    config = configparser.ConfigParser(defaults=defaults)
    configfile = open(config_filename, "r")
    config.readfp(configfile)

    toaddrs = config.get("Email", "To").split()
    fromaddr = config.get("Email", "From").split()
    toSMS = config.get("Text_from_email", "phoneEmail").split()
    pw = config.get("Email", "PW")
    runtime = config.get("Setup", "Runtime").split()
    pausetime = config.get("Setup", "Pausetime").split()


    if len(runtime) == 1:
        runtime.join('days')
    if len(pausetime) == 1:
        paustime.join('hours')

    if runtime[1] == 'seconds':
        factor = 1
    elif runtime[1] == 'minutes':
        factor = 60
    elif runtime[1] == 'hours':
        factor = 60*60
    elif runtime[1] == 'days':
        factor = 60*60*24
    elif runtime[1] == 'months':
        factor = 60*60*24*30.4
    elif runtime[1] == 'years':
        factor = 6060*60*24*365.25
    else:
        # assume days
        factor = 60*60*24

    max_uptime = int(runtime[0]) * factor

    if pausetime[1] == 'seconds':
        factor = 1
    elif pausetime[1] == 'minutes':
        factor = 60
    elif pausetime[1] == 'hours':
        factor = 60*60
    elif pausetime[1] == 'days':
        factor = 60*60*24
    elif pausetime[1] == 'months':
        factor = 60*60*24*30.4
    elif pausetime[1] == 'years':
        factor = 6060*60*24*365.25
    else:
        # assume hour
        factor = 60*60

    seconds_between_runs = int(pausetime[0]) * factor

    # put params in a dictionary
    params['fromaddr'] = str(fromaddr[0])
    params['toaddrs'] = str(toaddrs[0])
    params['toSMS'] = toSMS[0]
    params['PW'] = str(pw)
    params['runtime'] = max_uptime
    params['pausetime'] = seconds_between_runs

    params['best_trained_models_folder']    = config.get("Valuation", 'best_trained_models_folder')
    params['stockList']                     = config.get("Valuation", 'stockList')
    params['stockList_predict']             = config.get("Valuation", 'stockList_predict')
    params['first_history_index']           = int(config.get("Valuation", 'first_history_index'))
    params['num_stocks']                    = int(config.get("Valuation", 'num_stocks'))
    params['num_periods_history']           = int(config.get("Valuation", 'num_periods_history'))
    params['shortest_incr_range']           = np.array(config.get("Valuation", 'shortest_incr_range').split(',')).astype('int')
    params['longest_incr_range']            = np.array(config.get("Valuation", 'longest_incr_range').split(',')).astype('int')
    params['feature_map_factor_range']      = np.array(config.get("Valuation", 'feature_map_factor_range').split(',')).astype('int')

    params['model_filter']                      = config.get("Valuation", 'model_filter')
    params['sort_mode']                         = config.get("Valuation", 'sort_mode')
    params['folder_for_best_performers']        = config.get("Valuation", 'folder_for_best_performers')
    params['months_for_performance_comparison'] = int(config.get("Valuation", 'months_for_performance_comparison'))
    params['final_system_value_threshold']      = int(config.get("Valuation", 'final_system_value_threshold'))
    params['sharpe_threshold_percentile'] = int(config.get("Valuation", 'sharpe_threshold_percentile'))
    params['sortino_threshold_percentile']      = int(config.get("Valuation", 'sortino_threshold_percentile'))


    return params


def GetFTPParams():
    ######################
    ### Input FTP parameters
    ######################

    # set default values
    ftpparams = {}

    # read the parameters form the configuration file
    config_filename = "PyTAAADL.params"

    config = ConfigParser.ConfigParser()
    configfile = open(config_filename, "r")
    config.readfp(configfile)

    ftpHostname = config.get("FTP", "hostname")
    ftpUsername = config.get("FTP", "username")
    ftpPassword = config.get("FTP", "password")
    ftpRemotePath = config.get("FTP", "remotepath")
    ftpRemoteIP   = config.get("FTP", "remoteIP")

    # put params in a dictionary
    ftpparams['ftpHostname'] = str( ftpHostname )
    ftpparams['ftpUsername'] = str( ftpUsername )
    ftpparams['ftpPassword'] = str( ftpPassword )
    ftpparams['remotepath'] = str( ftpRemotePath )
    ftpparams['remoteIP'] = str( ftpRemoteIP )

    return ftpparams


def GetHoldings():
    ######################
    ### Input current holdings and cash
    ######################

    # set default values
    holdings = {}

    # read the parameters form the configuration file
    config_filename = "PyTAAADL_holdings.params"

    config = ConfigParser.ConfigParser()
    configfile = open(config_filename, "r")
    config.readfp(configfile)

    # put params in a dictionary
    holdings['stocks'] = config.get("Holdings", "stocks").split()
    holdings['shares'] = config.get("Holdings", "shares").split()
    holdings['buyprice'] = config.get("Holdings", "buyprice").split()
    holdings['cumulativecashin'] = config.get("Holdings", "cumulativecashin").split()

    # get rankings for latest dates for all stocks in index
    # read the parameters form the configuration file
    print(" ...inside GetHoldings...  pwd = ", os.getcwd())
    config_filename = "PyTAAADL_ranks.params"
    configfile = open(config_filename, "r")
    config.readfp(configfile)
    symbols = config.get("Ranks", "symbols").split()
    ranks = config.get("Ranks", "ranks").split()
    # put ranks params in dictionary
    holdings_ranks = []
    print("\n\n********************************************************")
    print(" ...inside GetParams/GetHoldings...")
    for i, holding in enumerate(holdings['stocks']):
        for j,symbol in enumerate(symbols):
            print("... j, symbol, rank = ", j, symbol, ranks[j])
            if symbol == holding:
                print("                                       MATCH ... i, symbol, rank = ", i, holding, symbols[j], ranks[j])
                holdings_ranks.append( ranks[j] )
                break
    holdings['ranks'] = holdings_ranks
    print("\n\n********************************************************")

    return holdings


def GetStatus():
    ######################
    ### Input current cumulative value
    ######################

    # read the parameters form the configuration file
    status_filename = "PyTAAADL_status.params"

    config = ConfigParser.ConfigParser()
    configfile = open(status_filename, "r")
    config.readfp(configfile)

    # put params in a dictionary
    status = config.get("Status", "cumu_value").split()[-3]

    return status

def PutStatus( cumu_status ):
    ######################
    ### Input current cumulative value
    ######################

    import datetime

    # read the parameters form the configuration file
    status_filename = "PyTAAADL_status.params"

    # check last value written to file for comparison with current cumu_status. Update if different.
    with open(status_filename, 'r') as f:
        lines = f.read()
    old_cumu_status = lines.split("\n")[-2]
    #old_cumu_status = old_cumu_status.split(" ")[-1]
    old_cumu_status = old_cumu_status.split(" ")[-3]

    old_cumu_signal = lines.split("\n")[-2]
    old_cumu_signal = old_cumu_signal.split(" ")[-2]

    # check current signal based on system protfolio value trend
    _, traded_values, _, last_signal = computeLongHoldSignal()

    print("cumu_status = ", str(cumu_status))
    print("old_cumu_status = ", old_cumu_status)
    print("last_signal[-1] = ", last_signal[-1])
    print("old_cumu_signal = ", old_cumu_signal)
    print(str(cumu_status)== old_cumu_status, str(last_signal[-1])== old_cumu_signal)
    if str(cumu_status) != str(old_cumu_status) or str(last_signal[-1]) != str(old_cumu_signal):
        with open(status_filename, 'a') as f:
            f.write( "cumu_value: "+\
                     str(datetime.datetime.now())+" "+\
                     str(cumu_status)+" "+\
                     str(last_signal[-1])+" "+\
                     str(traded_values[-1])+"\n" )

    return


def GetIP( ):
    ######################
    ### Input current cumulative value
    ######################

    import urllib
    import re
    f = urllib.urlopen("http://www.canyouseeme.org/")
    html_doc = f.read()
    f.close()
    m = re.search('(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',html_doc)
    return m.group(0)

def GetEdition( ):
    ######################
    ### Input current cumulative value
    ######################

    import platform

    # get edition from where software is running
    if 'armv6l' in platform.uname()[4] :
        edition = 'pi'
    elif 'x86' in platform.uname()[4] :
        edition = 'Windows32'
    elif 'AMD64' in platform.uname()[4] :
        edition = 'Windows64'
    else:
        edition = 'none'

    return edition
