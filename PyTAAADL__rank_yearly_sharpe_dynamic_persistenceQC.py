import os
import numpy as np
import datetime
import pandas

_cwd = os.getcwd()
#os.chdir(os.path.dirname(__file__))
from functions.allstats import allstats
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
os.chdir(_cwd)

def avg_DD(x,periods):
    DD = np.zeros_like(np.array(x))
    maxx = x[0]
    for i in range(1,len(DD)):
        maxx = max(maxx,x[i])
        DD[i] = (x[i]-maxx)/maxx
    return DD[-periods:].mean()

datearray_new_months = [datetime.date(1997, 1, 2)]
#df3 = pandas.HDFStore(os.path.join(_cwd,'pngs','best_performers4','persistence_data_full.hdf')).select('table')

persistence_values = []
persistence_number_stocks_choices = []
persistence_sorting_choices = []
unique_id_list = []
final_values = []
persistence_series_values = []
persistence_sortinos = []
persistence_mid_sortinos = []
persistence_recent_sortinos = []
persistence_sharpes = []
persistence_mid_sharpes = []
persistence_recent_sharpes = []
persistence_DD = []
persistence_mid_DD = []
persistence_recent_DD = []
persistence_months_list = range(10,20)
number_stocks_choices = [[2,7],[2,4,7],[2,5,7,9],[2,3,4,5,6,7,8,9],[2,5,6,7,8],[3,5,7,9],[2,4,6,8]]
sorting_choices = [['sortino','sharpe','count','equal'],['sortino','sharpe','count']]
#number_stocks_choices = [[2,7],[2,4,7]]

for inum_sort_choice in sorting_choices:
    for inum_stocks_choices in number_stocks_choices:
        df5 = pandas.DataFrame(columns=['dates', 'sort_modes', 'number_stocks', 'gains', 'symbols', 'weights', 'cumu_value'])
        for i,idate in enumerate(df3.index):
            datarow=list(df3.values[i])
            datarow.insert(0, idate)
            #print(i,df3.values[i,1])
            if i%1000==0:
                print("...progress... ",i)
            if df3.values[i,1] in inum_stocks_choices and df3.values[i,0] in inum_sort_choice:
                df5.loc[len(df5)] = datarow

        # generate (unique) lists with unique values for number_stocks, sort_modes, and dates
        _dates = df5.values[:,0]
        _number_stocks = df5.values[:,2]
        _sort_mode = df5.values[:,1]
        dates_list = list(set(_dates))
        number_stocks_list = list(set(_number_stocks))
        sort_mode_list = list(set(_sort_mode))
        dates_list.sort()
        number_stocks_list.sort()
        sort_mode_list.sort()

        for persistence_months in persistence_months_list:

            # how many months of persistence should be used to indicate 'best'?
            print("\n\n\n ... persistence_months = ", persistence_months, " ...\n\n")
            from time import sleep
            sleep(5)

            cumu_dynamic_system = [10000.0]
            plotdates = [datearray_new_months[0]]

            recent_comparative_gain = [1.]
            recent_comparative_month_gain = [1.]
            recent_comparative_method = ['cash']
            recent_comparative_nstocks = [0]

            for i, idate in enumerate(dates_list):

                recent_comparative_gain = [1.]
                recent_comparative_month_gain = [1.]
                recent_comparative_method = ['cash']
                recent_comparative_nstocks = [0]

                for inum_stocks in number_stocks_list:
                    for sort_mode in sort_mode_list:

                        indices = df5.loc[np.logical_and(df5['sort_modes']==sort_mode,df5['number_stocks']==inum_stocks)]['gains'].index
                        dates_for_selected = df5.values[indices,0]
                        cumugains_for_selected = 10000.*(df5.values[indices,-4]+1.).cumprod()
                        idate_index = np.argmin(np.abs(dates_for_selected - idate))
                        method_cumu_gains = cumugains_for_selected[:idate_index+1]

                        try:
                            recent_comparative_gain.append(method_cumu_gains[-2]/method_cumu_gains[-persistence_months-2])
                            recent_comparative_month_gain.append(method_cumu_gains[-1]/method_cumu_gains[-2])
                            recent_comparative_method.append(sort_mode)
                            recent_comparative_nstocks.append(inum_stocks)
                        except:
                            recent_comparative_gain.append(1.)
                            recent_comparative_month_gain.append(1.)
                            recent_comparative_method.append('cash')
                            recent_comparative_nstocks.append(0)

                        if sort_mode == sort_mode_list[-1] and inum_stocks == number_stocks_list[-1]:
                            plotdates.append(idate)
                            best_comparative_index = np.argmax(recent_comparative_gain)
                            cumu_dynamic_system.append(cumu_dynamic_system[-1] * recent_comparative_month_gain[best_comparative_index])
                            '''
                            print("        ... methods, near-term gains ",
                                  str(idate), recent_comparative_method,
                                  np.around(recent_comparative_gain,2))
                            '''
                            print("        ... dynamic system = ", str(idate),
                                  recent_comparative_nstocks[best_comparative_index],
                                  recent_comparative_method[best_comparative_index],
                                  format(cumu_dynamic_system[-1], '10,.0f'))

            persistence_sorting_choices.append(inum_sort_choice)
            persistence_number_stocks_choices.append(inum_stocks_choices)
            persistence_values.append(persistence_months)
            final_values.append(cumu_dynamic_system[-1])
            persistence_series_values.append(cumu_dynamic_system)
            persistence_DD.append(np.array(cumu_dynamic_system),len(cumu_dynamic_system))
            persistence_mid_DD.append(np.array(cumu_dynamic_system),int(12*10.75))
            persistence_recent_DD.append(np.array(cumu_dynamic_system),int(12*4.75))
            try:
                persistence_sortinos.append(allstats(np.array(cumu_dynamic_system)).sortino())
            except:
                persistence_sortinos.append(0.)
            try:
                persistence_mid_sortinos.append(allstats(np.array(cumu_dynamic_system[-int(12*10.75):])).sortino())
            except:
                persistence_mid_sortinos.append(0.)
            try:
                persistence_recent_sortinos.append(allstats(np.array(cumu_dynamic_system[-int(12*4.75):])).sortino())
            except:
                persistence_recent_sortinos.append(0.)
            try:
                persistence_sharpes.append(allstats(np.array(cumu_dynamic_system)).monthly_sharpe())
            except:
                persistence_sharpes.append(0.)
            try:
                persistence_mid_sharpes.append(allstats(np.array(cumu_dynamic_system[-int(12*10.75):])).monthly_sharpe())
            except:
                persistence_mid_sharpes.append(0.)
            try:
                persistence_recent_sharpes.append(allstats(np.array(cumu_dynamic_system[-int(12*4.75):])).monthly_sharpe())
            except:
                persistence_recent_sharpes.append(0.)

            unique_id = str(persistence_months)+str(inum_stocks_choices)+str(inum_sort_choice)
            unique_id_list.append(unique_id)

            if persistence_months == persistence_months_list[0] and inum_stocks_choices == number_stocks_choices[0] and inum_sort_choice == sorting_choices[0]:
                plt.close(1)
                plt.figure(1)
                plt.yscale('log')
                plt.grid(True)
            if persistence_months==18:
                plt.plot(plotdates, cumu_dynamic_system, 'k-', lw=3.5, label=str(persistence_months)+str(inum_stocks_choices))
            elif persistence_months==16:
                plt.plot(plotdates, cumu_dynamic_system, 'y-', lw=3.5, label=str(persistence_months)+str(inum_stocks_choices))
            elif persistence_months==19:
                plt.plot(plotdates, cumu_dynamic_system, 'r-', lw=3.5, label=str(persistence_months)+str(inum_stocks_choices))
            else:
                plt.plot(plotdates, cumu_dynamic_system, label=str(persistence_months)+str(inum_stocks_choices))

plt.legend()

plt.close(2)
plt.figure(2)
plt.grid(True)
'''
plt.plot(persistence_values,final_values/np.max(final_values),label='final_value')
plt.plot(persistence_values,persistence_sortinos,label='sortino')
plt.plot(persistence_values,persistence_mid_sortinos,label='mid_sortino')
plt.plot(persistence_values,persistence_recent_sortinos,label='recent_sortino')
plt.plot(persistence_values,persistence_sharpes,'--',label='sharpe')
plt.plot(persistence_values,persistence_mid_sharpes,'--',label='mid_sharpe')
plt.plot(persistence_values,persistence_recent_sharpes,'--',label='recent_sharpe')
'''
plt.plot(range(len(unique_id_list)),final_values/np.max(final_values),label='final_value')
plt.plot(range(len(unique_id_list)),persistence_sortinos,label='sortino')
plt.plot(range(len(unique_id_list)),persistence_mid_sortinos,label='mid_sortino')
plt.plot(range(len(unique_id_list)),persistence_recent_sortinos,label='recent_sortino')
plt.plot(range(len(unique_id_list)),persistence_sharpes,'--',label='sharpe')
plt.plot(range(len(unique_id_list)),persistence_mid_sharpes,'--',label='mid_sharpe')
plt.plot(range(len(unique_id_list)),persistence_recent_sharpes,'--',label='recent_sharpe')
plt.legend()

plt.close(3)
plt.figure(3)
plt.clf()
plt.grid(True)
plt.yscale('log')

id = np.argmax(final_values)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'final_value')
id = np.argsort(final_values)
sum_id_ranks = (id/10.).astype('int')
weighted_sum_id_ranks = (id/10.).astype('int')*4

id = np.argmax(persistence_sortinos)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_sortinos')
id = np.argsort(persistence_sortinos)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')

id = np.argmax(persistence_mid_sortinos)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_mid_sortinos')
id = np.argsort(persistence_mid_sortinos)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')*2

id = np.argmax(persistence_recent_sortinos)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_recent_sortinos')
id = np.argsort(persistence_recent_sortinos)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')*3

id = np.argmax(persistence_sharpes)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_sharpes')
id = np.argsort(persistence_sharpes)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')

id = np.argmax(persistence_mid_sharpes)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_mid_sharpes')
id = np.argsort(persistence_mid_sharpes)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')*2

id = np.argmax(persistence_recent_sharpes)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_recent_sharpes')
id = np.argsort(persistence_recent_sharpes)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')*3

id = np.argmax(persistence_DD)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_DD')
id = np.argsort(persistence_DD)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')

id = np.argmax(persistence_mid_DD)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_mid_DD')
id = np.argsort(persistence_mid_DD)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')*2

id = np.argmax(persistence_recent_DD)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'persistence_recent_DD')
id = np.argsort(persistence_recent_DD)
sum_id_ranks += (id/10.).astype('int')
weighted_sum_id_ranks += (id/10.).astype('int')*3

id = np.argmax(sum_id_ranks)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'summed ranks')

id = np.argmax(weighted_sum_id_ranks)
plt.plot(plotdates,persistence_series_values[id],label=unique_id_list[id]+'weighted sum ranks')

plt.legend()

