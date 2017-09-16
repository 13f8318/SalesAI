import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.path as mpath
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
import os
import csv
from os import listdir
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import copy
import operator
import timeit
start = timeit.default_timer()

df_logs = pd.DataFrame()
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id', 'seller_id', 'action_type'])
print('Length of df_logs:' , len(df_logs))

#---------------------------------------------- Features: 70, 71, 72, 73 Start-----------------------------------------------

temporary_df_clicks = df_logs[(df_logs.action_type == 0)]
temporary_df_clicks=temporary_df_clicks.groupby(['user_id', 'seller_id']).size().rename('count_clicks').reset_index()
print('Total Clicks : ', len(temporary_df_clicks))

temporary_df_cart = pd.DataFrame(df_logs[(df_logs.action_type == 1)].groupby(['user_id', 'seller_id']).size().rename('count_carts').reset_index())
print('Total Carts : ', len(temporary_df_cart))

temporary_df_buy = pd.DataFrame(df_logs[(df_logs.action_type == 2)].groupby(['user_id', 'seller_id']).size().rename('count_buys').reset_index())
print('Total Buys : ', len(temporary_df_buy))

temporary_df_fav = pd.DataFrame(df_logs[(df_logs.action_type == 3)].groupby(['user_id', 'seller_id']).size().rename('count_favs').reset_index())
print('Total Favs : ', len(temporary_df_fav))

#print('-------------- Merge Click Cart-----------------')
merge_click_cart = pd.DataFrame()
merge_click_cart = temporary_df_clicks.merge(temporary_df_cart,how='left', on=["user_id","seller_id"]).fillna(0)

#print('--------------Merge Click Cart Buy-------------')
merge_click_cart_buy = pd.DataFrame()
merge_click_cart_buy = merge_click_cart.merge(temporary_df_buy,how='left', on=["user_id","seller_id"]).fillna(0)

#print('--------------Merge Click Cart Buy Fav----------')
merge_click_cart_buy_fav = pd.DataFrame()
merge_click_cart_buy_fav = merge_click_cart_buy.merge(temporary_df_fav,how='left', on=["user_id","seller_id"]).fillna(0)

del temporary_df_clicks
#Here extract those rows of temporary_df_cart that went to merge_click_cart and delete those from temprary_df_cart
#and then save them to remaining_carts.
extract_from_cc = pd.DataFrame
extract_from_cc = merge_click_cart[(merge_click_cart.count_carts != 0)]
del extract_from_cc['count_clicks']
del merge_click_cart

remaining_carts = pd.DataFrame
remaining_carts = pd.concat([temporary_df_cart, extract_from_cc]).drop_duplicates(keep=False)
remaining_carts = remaining_carts.reset_index(drop=True)
del extract_from_cc
del temporary_df_cart
print('1 Remaining Carts: ', len(remaining_carts))

#Here extract those rows of temporary_df_buy that went to merge_click_cart_buy and delete those from temprary_df_buy
#and then save them to remaining_buys.
extract_from_ccb = pd.DataFrame
extract_from_ccb = merge_click_cart_buy[(merge_click_cart_buy.count_buys!=0)]
del extract_from_ccb['count_clicks'], extract_from_ccb['count_carts']
del merge_click_cart_buy

remaining_buys = pd.DataFrame()
remaining_buys = pd.concat([temporary_df_buy, extract_from_ccb]).drop_duplicates(keep=False)
remaining_buys = remaining_buys.reset_index(drop=True)
print('1 Remaining Buys: ', len(remaining_buys))
del extract_from_ccb
del temporary_df_buy

#Here extract those rows of temporary_df_fav that went to merge_click_cart_buy_fav and delete those from temprary_df_fav
#and then save them to remaining_favs.
extract_from_ccbf = pd.DataFrame
extract_from_ccbf = merge_click_cart_buy_fav[(merge_click_cart_buy_fav.count_favs!=0)]
del extract_from_ccbf['count_clicks'], extract_from_ccbf['count_carts'], extract_from_ccbf['count_buys']
#del merge_click_cart_buy_fav

remaining_favs = pd.DataFrame()
remaining_favs = pd.concat([temporary_df_fav, extract_from_ccbf]).drop_duplicates(keep=False)
remaining_favs = remaining_favs.reset_index(drop=True)
print('1 Remaining Favs: ', len(remaining_favs))
del extract_from_ccbf
del temporary_df_fav

#------------------------------------------------------Merge2-----------------------------------------------------------
#Now we will merge remaining carts and remaining buys.
merge_rcb = pd.DataFrame()
merge_rcb = remaining_carts.merge(remaining_buys,how='left', on=["user_id","seller_id"]).fillna(0)
del remaining_carts

#Now merge merge_rcb and remaining_favs
merge_rcb_rf = pd.DataFrame()
merge_rcb_rf = merge_rcb.merge(remaining_favs,how='left', on=["user_id","seller_id"]).fillna(0)

fill_missing = np.zeros(len(merge_rcb_rf))
merge_rcb_rf.insert(2, "count_clicks", fill_missing)

print('Inital Length of merge_click_cart_buy_fav: ', len(merge_click_cart_buy_fav))
merge_click_cart_buy_fav = merge_click_cart_buy_fav.append(merge_rcb_rf, ignore_index = True)
print('Length After Merge 2: ', len(merge_click_cart_buy_fav))

#----------------------------------------------------------------------------------------------------------------

#Here extract those rows of remaining_buys that went to merge_rcb and delete those from remaining_buys
#and then save them to remaining_buys_merge2
extract_from_rcb = pd.DataFrame()
extract_from_rcb = merge_rcb[(merge_rcb.count_buys != 0)]
del extract_from_rcb['count_carts']
del merge_rcb

remaining_buys_merge2 = pd.DataFrame()
remaining_buys_merge2 = pd.concat([remaining_buys, extract_from_rcb]).drop_duplicates(keep=False)
remaining_buys_merge2 = remaining_buys_merge2.reset_index(drop=True)
del extract_from_rcb
del remaining_buys

#Here extract those rows of remaining_favs that went to merge_rcb_rf and delete those from remaining_favs
#and then save them to remaining_favs_merge2.
extract_from_rcb_rf = pd.DataFrame()
extract_from_rcb_rf = merge_rcb_rf[(merge_rcb_rf.count_favs != 0)]
del extract_from_rcb_rf['count_clicks'], extract_from_rcb_rf['count_carts'], extract_from_rcb_rf['count_buys']

remaining_favs_merge2 = pd.DataFrame()
remaining_favs_merge2 = pd.concat([remaining_favs, extract_from_rcb_rf]).drop_duplicates(keep=False)
remaining_favs_merge2 = remaining_favs_merge2.reset_index(drop=True)
del extract_from_rcb_rf
del remaining_favs

print('2 Length of remaining_Buys_merge2: ', len(remaining_buys_merge2))
print('2 Length of remaining_Favs_merge2: ', len(remaining_favs_merge2))
#
#----------------------------------------Merge 3------------------------------------------------------------------------
#Now we will merge remaining buys_merge2 and remaining_favs_merge2
merge_rbf = pd.DataFrame()
merge_rbf = remaining_buys_merge2.merge(remaining_favs_merge2,how='left', on=["user_id","seller_id"]).fillna(0)
del remaining_buys_merge2

fill_missing = np.zeros(len(merge_rbf ))
merge_rbf.insert(2, "count_clicks", fill_missing)
merge_rbf.insert(3, "count_carts", fill_missing)

merge_click_cart_buy_fav = merge_click_cart_buy_fav.append(merge_rbf, ignore_index= True)
print('Length After Merge 3: ', len(merge_click_cart_buy_fav))

#-----------------------------------------------------------------------------------------------------------------------

#Here extract those rows of remaining_favs_merge2 that went to merge_rbf and delete those from remaining_favs_merge2
#and then save them to remaining_favs_merge3.
extract_from_rbf = pd.DataFrame()
extract_from_rbf = merge_rbf[(merge_rbf.count_favs != 0)]
del extract_from_rbf['count_buys'], extract_from_rbf['count_clicks'], extract_from_rbf['count_carts']
#del merge_rbf

remaining_favs_merge3 = pd.DataFrame()
remaining_favs_merge3 = pd.concat([remaining_favs_merge2, extract_from_rbf]).drop_duplicates(keep=False)
remaining_favs_merge3 = remaining_favs_merge3.reset_index(drop=True)
del extract_from_rbf
del remaining_favs_merge2

print('3 Length of Remaining Favs Merge 3: ', len(remaining_favs_merge3))

#----------------------------------------------- Last Appending -------------------------------------------------------

list_p = np.zeros(len(remaining_favs_merge3))
remaining_favs_merge3.insert(2, "count_clicks", list_p)
remaining_favs_merge3.insert(3, "count_carts", list_p)
remaining_favs_merge3.insert(4, "count_buys", list_p)

merge_click_cart_buy_fav = merge_click_cart_buy_fav.append(remaining_favs_merge3, ignore_index = True)
print('Length After Last Merge: ', len(merge_click_cart_buy_fav))

merge_click_cart_buy_fav = merge_click_cart_buy_fav.sort_values(['user_id'], ascending=True)
merge_click_cart_buy_fav.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv', index=False)

#---------------------------------------------- Features: 70, 71, 72, 73 END--------------------------------------------

df_features = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv', skipinitialspace=True, usecols=['user_id', 'seller_id'])

#--------------------------------------Feature 74: LIFE SPAN Start------------------------------------------------------
print('Feature 74: Started')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id', 'seller_id', 'time_stamp'])
df_logs = df_logs.sort_values(['user_id'],ascending=True)

#print('Merging Features and Logs DataFrames.')
merge_fl = df_features.merge(df_logs,how='left', on=["user_id","seller_id"])
merge_fl = merge_fl.drop_duplicates()

#Merge1 contains the maximum timestamp value for each unique customer and seller pair.
get_max = merge_fl.groupby(['user_id', 'seller_id']).time_stamp.transform(max)
merge1 = merge_fl[merge_fl.time_stamp == get_max]
del get_max

#Merge2 contains the minimum timestamp value for each unique customer and seller p
get_min = merge_fl.groupby(['user_id', 'seller_id']).time_stamp.transform(min)
merge2 = merge_fl[merge_fl.time_stamp == get_min]
del merge_fl

#Now Merge both Merge1 and Merge2 so now Lifespan will contain one column for min timestamp and other column for max
#timestamp for each unique customer and seller pair.
life_span = merge2.merge(merge1,how='left', on=["user_id","seller_id"])
del merge1, merge2, df_logs

life_span['life_span'] = life_span["time_stamp_x"].map(str) + '_' +  life_span["time_stamp_y"].map(str)
del life_span["time_stamp_x"], life_span["time_stamp_y"]

df = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv')
new_column = pd.DataFrame({'life_span': life_span['life_span']})
df = df.merge(new_column, left_index = True, right_index = True)
df.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv', index=False)
del life_span
print('Feature 74: Completed')
#------------------------------------------ Feature 74: End---------------------------------------------------------------

#----------------Feature 75: The number of different items that a user has behaviour to a merchant ----------------------
print('Feature 75: Started')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id', 'seller_id', 'item_id'])
df_logs = df_logs.sort_values(['user_id'],ascending=True)

unique = df_logs.drop_duplicates(subset=['user_id', 'item_id', 'seller_id'])
unique = unique.sort_values(['user_id'],ascending=True)

unique = unique.groupby(["user_id", "seller_id"]).size().reset_index(name="unique_items")
unique = unique.sort_values(['user_id'],ascending=True)

merge = df_features.merge(unique, how="left",  on=['user_id', 'seller_id']).fillna(0)

del df_logs, unique
df = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv')
new_column = pd.DataFrame({'Unique Items Behavior': merge['unique_items']})
df = df.merge(new_column, left_index = True, right_index = True)
df.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv', index=False)
del merge
print('Feature 75: Completed')

#-----------------------------------------------Feature 75: END---------------------------------------------------------

#-----------------Feature 76: The number of different categories that a user has behaviour to a merchant---------------
print('Feature 76: Started')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id', 'seller_id', 'cat_id'])
df_logs = df_logs.sort_values(['user_id'],ascending=True)

unique = df_logs.drop_duplicates(subset=['user_id', 'cat_id', 'seller_id'], keep='first')
unique = unique.sort_values(['user_id'],ascending=True)

unique = unique.groupby(["user_id", "seller_id"]).size().reset_index(name="unique_cat")
unique = unique.sort_values(['user_id'],ascending=True)

merge = df_features.merge(unique, how="left",  on=['user_id', 'seller_id']).fillna(0)

del df_logs, unique
df = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv')
new_column = pd.DataFrame({'unique cat': merge['unique_cat']})
df = df.merge(new_column, left_index = True, right_index = True)
df.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv', index=False)
del merge
print('Feature 76: Completed')

#---------------------------------------------------Feature: 76 End-----------------------------------------------------

#----Feature 77: the interactive number of brand that a user has ever bought and a merchant has ever sold --------------
print('Feature 77: Started')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id', 'seller_id','brand_id', 'action_type'])
df_logs = df_logs.sort_values(['user_id'],ascending=True)

df_logs = df_logs[df_logs.action_type == 2]

df_logs = df_logs.drop_duplicates()
del df_logs['action_type']

df_logs = df_logs.groupby(['user_id', 'seller_id']).size().reset_index(name = 'F77')

merge_fl = pd.DataFrame() #Merge_fl = Merge features and log
merge_fl = df_features.merge(df_logs, how="left", on= ["user_id", "seller_id"]).fillna(0)

del df_logs
df = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv')
new_column = pd.DataFrame({'BUBMS:F77': merge_fl['F77']})
df = df.merge(new_column, left_index = True, right_index = True)
df.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv', index=False)
del merge_fl
print('Feature 77: Completed')

#--------------------------------------------------- Feature 77: END ---------------------------------------------------

#--------Feature 78: the interactive number of Category that a user has ever bought and a merchant has ever sold -------
print('Feature 78: Started')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id', 'seller_id','cat_id', 'action_type'])
df_logs = df_logs.sort_values(['user_id'],ascending=True)

df_logs = df_logs[df_logs.action_type == 2]

df_logs = df_logs.drop_duplicates()
del df_logs['action_type']

df_logs = df_logs.groupby(['user_id', 'seller_id']).size().reset_index(name = 'F78')

merge_fl = pd.DataFrame() #Merge_fl = Merge features and log
merge_fl = df_features.merge(df_logs, how="left", on= ["user_id", "seller_id"]).fillna(0)

del df_logs
df = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv')
new_column = pd.DataFrame({'CUBSS:F78': merge_fl['F78']})
df = df.merge(new_column, left_index = True, right_index = True)
df.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv', index=False)
del merge_fl
print('Feature 78: Completed')
#--------------------------------------------------- Feature 78: END ---------------------------------------------------
stop = timeit.default_timer()

print('Total Run Time')
print((stop - start) , ' Seconds.')
print(((stop - start)/60) , ' Minutes.')

