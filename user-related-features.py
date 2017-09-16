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

df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id', 'action_type', 'time_stamp'])

age_gender = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_info_format1.csv', skipinitialspace=True, usecols=['user_id', 'age_range', 'gender'])
age_gender = age_gender.drop_duplicates()
age_gender = age_gender.sort_values(['user_id'], ascending=True)


print('3, 4 , 5, 6')
#--------------------------------------------- Feature: 3, 4, 5, 6 Start------------------------------------------------

click_times = df_logs[(df_logs.action_type == 0) & (df_logs.time_stamp!= 1111)].groupby('user_id').size().reset_index(name='click_times')
click_times = click_times.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(click_times, how="left", on=["user_id"]).fillna(0) #merge unique users and click times.
del click_times


cart_times = df_logs[(df_logs.action_type == 1) & (df_logs.time_stamp!= 1111)].groupby('user_id').size().reset_index(name='cart_times')
cart_times = cart_times.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(cart_times, how="left", on=["user_id"]).fillna(0)
del cart_times

buy_times = df_logs[(df_logs.action_type == 2) & (df_logs.time_stamp!= 1111)].groupby('user_id').size().reset_index(name='buy_times')
buy_times = buy_times.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(buy_times, how="left", on=["user_id"]).fillna(0)
del buy_times


fav_times = df_logs[(df_logs.action_type == 3) & (df_logs.time_stamp!= 1111)].groupby('user_id').size().reset_index(name='fav_times')
fav_times = fav_times.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(fav_times, how="left", on=["user_id"]).fillna(0)
del fav_times

#--------------------------------------------- Feature: 3, 4, 5, 6 End--------------------------------------------------

#---------------------- F:7 The number of different days that a user have click behaviour-------------------------------

print('7')
click_behavior = df_logs[df_logs.action_type == 0]
click_behavior = click_behavior.drop_duplicates()
diff_days = click_behavior.groupby('user_id').size().reset_index(name="no_of_diffdays_click_behav")
diff_days = diff_days.sort_values(['user_id'], ascending=True)

age_gender = age_gender.merge(diff_days, how="left", on=["user_id"]).fillna(0)

del click_behavior, diff_days

#---------------------- F:8 The number of different days that a user have buy behaviour---------------------------------
print('8')
buy_behavior = df_logs[df_logs.action_type == 2]
buy_behavior = buy_behavior.drop_duplicates()
diff_days = buy_behavior.groupby('user_id').size().reset_index(name="no_of_diffdays_buy_behav")
diff_days = diff_days.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_days, how="left", on=["user_id"]).fillna(0)
del buy_behavior, diff_days

#----------------------F:9 the number of different items that a user click----------------------------------------------
print('9')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id','item_id','action_type'])
total_clicks = df_logs[df_logs.action_type == 0]
diff_clicks = total_clicks.drop_duplicates()
del total_clicks
diff_clicks= diff_clicks.groupby('user_id').size().reset_index(name="diff_items_clicked")
diff_clicks = diff_clicks.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_clicks, how="left", on=["user_id"]).fillna(0)
del diff_clicks

#-----------------------F:10 the number of different items that a user add to cart--------------------------------------
print('10')
total_carts = df_logs[df_logs.action_type == 1]
diff_carts = total_carts.drop_duplicates()
del total_carts
diff_carts= diff_carts.groupby('user_id').size().reset_index(name="diff_items_carted")
diff_carts = diff_carts.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_carts, how="left", on=["user_id"]).fillna(0)
del diff_carts

#------------------------------F:11 the number of different items that a user buy---------------------------------------
print('11')
total_buys = df_logs[df_logs.action_type == 2]
diff_buys = total_buys.drop_duplicates()
del total_buys
diff_buys= diff_buys.groupby('user_id').size().reset_index(name="diff_items_bought")
diff_buys= diff_buys.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_buys, how="left", on=["user_id"]).fillna(0)
del diff_buys

#------------------------------F:12 the number of different items that a user add to fav--------------------------------
print('12')
total_favs = df_logs[df_logs.action_type == 3]
diff_favs = total_favs.drop_duplicates()
del total_favs
diff_favs= diff_favs.groupby('user_id').size().reset_index(name="diff_items_favorited")
diff_favs= diff_favs.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_favs, how="left", on=["user_id"]).fillna(0)
del diff_favs

#-------------------F:13 the number of different category that a user click---------------------------------------------
print('13')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id','cat_id','action_type'])
total_clicks = df_logs[df_logs.action_type == 0]
diff_clicks = total_clicks.drop_duplicates()
del total_clicks
diff_clicks= diff_clicks.groupby('user_id').size().reset_index(name="diff_cat_clicked")
diff_clicks= diff_clicks.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_clicks, how="left", on=["user_id"]).fillna(0)
del diff_clicks

#-----------------------F:14 the number of different category that a user add to cart-----------------------------------
print('14')
total_carts = df_logs[df_logs.action_type == 1]
diff_carts = total_carts.drop_duplicates()
del total_carts
diff_carts= diff_carts.groupby('user_id').size().reset_index(name="diff_cat_carted")
diff_carts= diff_carts.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_carts, how="left", on=["user_id"]).fillna(0)
del diff_carts

#------------------F:15 The number of different category that a user buy------------------------------------------------
print('15')
total_buys = df_logs[df_logs.action_type == 2]
diff_buys = total_buys.drop_duplicates()
del total_buys
diff_buys= diff_buys.groupby('user_id').size().reset_index(name="diff_cat_bought")
diff_buys= diff_buys.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_buys, how="left", on=["user_id"]).fillna(0)
del diff_buys

#---------------F:16 the number of different category that a user add to fav--------------------------------------------
print('16')
total_favs = df_logs[df_logs.action_type == 3]
diff_favs = total_favs.drop_duplicates()
del total_favs
diff_favs= diff_favs.groupby('user_id').size().reset_index(name="diff_cat_favorited")
diff_favs= diff_favs.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_favs, how="left", on=["user_id"]).fillna(0)
del diff_favs

#----------- F:17 the number of different brands that a user click------------------------------------------------------
print('17')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id','brand_id','action_type'])
total_clicks = df_logs[df_logs.action_type == 0]
diff_clicks = total_clicks.drop_duplicates()
del total_clicks
diff_clicks= diff_clicks.groupby('user_id').size().reset_index(name="diff_brands_clicked")
diff_clicks= diff_clicks.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_clicks, how="left", on=["user_id"]).fillna(0)
del diff_clicks

#----------- F:18 the number of different brands that a user add to cart------------------------------------------------
print('18')
total_carts = df_logs[df_logs.action_type == 1]
diff_carts = total_carts.drop_duplicates()
del total_carts
diff_carts= diff_carts.groupby('user_id').size().reset_index(name="diff_brands_carted")
diff_carts= diff_carts.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_carts, how="left", on=["user_id"]).fillna(0)
del diff_carts

#--------------F:19 the number of different brands that a user buy------------------------------------------------------
print('19')
total_buys = df_logs[df_logs.action_type == 2]
diff_buys = total_buys.drop_duplicates()
del total_buys
diff_buys= diff_buys.groupby('user_id').size().reset_index(name="diff_brands_bought")
diff_buys= diff_buys.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_buys, how="left", on=["user_id"]).fillna(0)
del diff_buys

#---------------F:20  number of different brands that a user add to fav-------------------------------------------------
print('20')
total_favs = df_logs[df_logs.action_type == 3]
diff_favs = total_favs.drop_duplicates()
del total_favs
diff_favs= diff_favs.groupby('user_id').size().reset_index(name="diff_brands_favorited")
diff_favs= diff_favs.sort_values(['user_id'], ascending=True)
age_gender = age_gender.merge(diff_favs, how="left", on=["user_id"]).fillna(0)
del diff_favs

#-----------------------------Calculate Time for Count Features---------------------------------------------------------
stop = timeit.default_timer()
print('Total time to calculate Count Features: ' , (stop - start) , ' Seconds.')
print('Total time to calculate Count Features: ' ,((stop - start)/60) , ' Minutes.')

print('User Ratio Features Started.')
#---------------------- User Ratio Features ----------------------------------------------------------------------------
#------------------------------------ F:22 -----------------------------------------------------------------------------
print('22')
start = timeit.default_timer()

df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id','item_id','action_type', 'time_stamp'])
#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_items_clicked'])
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_items_clicked'] = age_gender['diff_items_clicked']
clicks_in11 = df_logs[(df_logs.action_type == 0) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_clicks_11 = clicks_in11.drop_duplicates(subset=['item_id'])
diff_clicks_11 = diff_clicks_11.groupby('user_id').size().reset_index(name='click_in11')

merge_both = users.merge(diff_clicks_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_items_click_in11'] = np.where(merge_both['diff_items_clicked'] < 1, merge_both['diff_items_clicked'], merge_both['click_in11']/merge_both['diff_items_clicked'])

age_gender = age_gender.merge(merge_both[['user_id', 'ratio_items_click_in11']], how="left", on=["user_id"]).fillna(0)
# age_gender.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/user-related-features.csv', index=False)
#
# stop = timeit.default_timer()
# print('Total Time: ' , (stop - start) , ' Seconds.')
# print('Total Time: ' ,((stop - start)/60) , ' Minutes.')

# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_items_click_d11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
#
# #-------------------------------- F:23 -------------------------------------------------------------------------------
#
#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_items_carted'])
print('23')
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_items_carted'] = age_gender['diff_items_carted']
carts_in11 = df_logs[(df_logs.action_type == 1) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_carts_11 = carts_in11.drop_duplicates(subset=['item_id'])
diff_carts_11 = diff_carts_11.groupby('user_id').size().reset_index(name='cart_in11')

merge_both = users.merge(diff_carts_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_items_cart_in11'] = np.where(merge_both['diff_items_carted'] < 1, merge_both['diff_items_carted'], merge_both['cart_in11']/merge_both['diff_items_carted'])

age_gender = age_gender.merge(merge_both[['user_id', 'ratio_items_cart_in11']], how="left", on=["user_id"]).fillna(0)

# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_items_cart_d11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
#
# #-------------------------------- F:24 --------------------------------------------
#
#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_items_bought'])
print('24')
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_items_bought'] = age_gender['diff_items_bought']

buys_in11 = df_logs[(df_logs.action_type == 2) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_buys_11 = buys_in11.drop_duplicates(subset=['item_id'])
diff_buys_11 = diff_buys_11.groupby('user_id').size().reset_index(name='buy_in11')

merge_both = users.merge(diff_buys_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_items_buy_in11'] = np.where(merge_both['diff_items_bought'] < 1, merge_both['diff_items_bought'], merge_both['buy_in11']/merge_both['diff_items_bought'])

age_gender = age_gender.merge(merge_both[['user_id', 'ratio_items_buy_in11']], how="left", on=["user_id"]).fillna(0)
# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_items_buy_d11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
#
# #------------------------------- F:25 -----------------------------------------------
#
#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_items_favorited'])
print('25')
favs_in11 = df_logs[(df_logs.action_type == 3) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_items_favorited'] = age_gender['diff_items_favorited']

diff_favs_11 = favs_in11.drop_duplicates(subset=['item_id'])
diff_favs_11 = diff_favs_11.groupby('user_id').size().reset_index(name='fav_in11')

merge_both = users.merge(diff_favs_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_items_fav_in11'] = np.where(merge_both['diff_items_favorited'] < 1, merge_both['diff_items_favorited'], merge_both['fav_in11']/merge_both['diff_items_favorited'])

age_gender = age_gender.merge(merge_both[['user_id', 'ratio_items_fav_in11']], how="left", on=["user_id"]).fillna(0)
# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_items_fav_d11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)

#------------------------------------ F:26 ---------------------------------------
print('26')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id','cat_id','action_type', 'time_stamp'])
#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_cat_clicked'])
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_cat_clicked'] = age_gender['diff_cat_clicked']

clicks_in11 = df_logs[(df_logs.action_type == 0) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_clicks_11 = clicks_in11.drop_duplicates(subset=['cat_id'])
diff_clicks_11 = diff_clicks_11.groupby('user_id').size().reset_index(name='click_in11')

merge_both = users.merge(diff_clicks_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_cat_click_in11'] = np.where(merge_both['diff_cat_clicked'] < 1, merge_both['diff_cat_clicked'], merge_both['click_in11']/merge_both['diff_cat_clicked'])
age_gender = age_gender.merge(merge_both[['user_id', 'ratio_cat_click_in11']], how="left", on=["user_id"]).fillna(0)

# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_cat_click_in11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
del merge_both, diff_clicks_11, clicks_in11
#
# #-------------------------------- F:27 --------------------------------------------
#
print('27')
#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_cat_carted'])
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_cat_carted'] = age_gender['diff_cat_carted']

carts_in11 = df_logs[(df_logs.action_type == 1) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_carts_11 = carts_in11.drop_duplicates(subset=['cat_id'])
diff_carts_11 = diff_carts_11.groupby('user_id').size().reset_index(name='cart_in11')

merge_both = users.merge(diff_carts_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_cat_cart_in11'] = np.where(merge_both['diff_cat_carted'] < 1, merge_both['diff_cat_carted'], merge_both['cart_in11']/merge_both['diff_cat_carted'])

age_gender = age_gender.merge(merge_both[['user_id', 'ratio_cat_cart_in11']], how="left", on=["user_id"]).fillna(0)

# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_cat_cart_in11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
del merge_both, diff_carts_11, carts_in11
#
# #-------------------------------- F:28 --------------------------------------------
#
print('28')
#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_cat_bought'])
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_cat_bought'] = age_gender['diff_cat_bought']

buys_in11 = df_logs[(df_logs.action_type == 2) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_buy_11 = buys_in11.drop_duplicates(subset=['cat_id'])
diff_buy_11 = diff_buy_11.groupby('user_id').size().reset_index(name='buy_in11')

merge_both = users.merge(diff_buy_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_cat_buy_in11'] = np.where(merge_both['diff_cat_bought'] < 1, merge_both['diff_cat_bought'], merge_both['buy_in11']/merge_both['diff_cat_bought'])

age_gender = age_gender.merge(merge_both[['user_id', 'ratio_cat_buy_in11']], how="left", on=["user_id"]).fillna(0)


# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_cat_buy_in11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
del merge_both, diff_buy_11, buys_in11

#
# #------------------------------- F:29 -----------------------------------------------
#
print('29')
# users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_cat_favorited'])
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_cat_favorited'] = age_gender['diff_cat_favorited']

favs_in11 = df_logs[(df_logs.action_type == 3) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_fav_11 = favs_in11.drop_duplicates(subset=['cat_id'])
diff_fav_11 = diff_fav_11.groupby('user_id').size().reset_index(name='fav_in11')

merge_both = users.merge(diff_fav_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_cat_fav_in11'] = np.where(merge_both['diff_cat_favorited'] < 1, merge_both['diff_cat_favorited'], merge_both['fav_in11']/merge_both['diff_cat_favorited'])

age_gender = age_gender.merge(merge_both[['user_id', 'ratio_cat_fav_in11']], how="left", on=["user_id"]).fillna(0)
# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_cat_fav_in11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
del merge_both, diff_fav_11, favs_in11

#------------------------------------ F:30 ---------------------------------------

print('30')
df_logs = pd.read_csv('/home/hassan/Desktop/SalesAI2/user_log_format1.csv', skipinitialspace=True, usecols=['user_id','brand_id','action_type', 'time_stamp'])
#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_brands_clicked'])
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_brands_clicked'] = age_gender['diff_brands_clicked']

clicks_in11 = df_logs[(df_logs.action_type == 0) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_clicks_11 = clicks_in11.drop_duplicates(subset=['brand_id'])
diff_clicks_11 = diff_clicks_11.groupby('user_id').size().reset_index(name='click_in11')

merge_both = users.merge(diff_clicks_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_brand_click_in11'] = np.where(merge_both['diff_brands_clicked'] < 1, merge_both['diff_brands_clicked'], merge_both['click_in11']/merge_both['diff_brands_clicked'])
age_gender = age_gender.merge(merge_both[['user_id', 'ratio_brand_click_in11']], how="left", on=["user_id"]).fillna(0)

# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_brand_click_in11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
del merge_both, diff_clicks_11, clicks_in11
#
# #-------------------------------- F:31 --------------------------------------------
#
# users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_brands_carted'])
print('31')
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_brands_carted'] = age_gender['diff_brands_carted']

carts_in11 = df_logs[(df_logs.action_type == 1) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_carts_11 = carts_in11.drop_duplicates(subset=['brand_id'])
diff_carts_11 = diff_carts_11.groupby('user_id').size().reset_index(name='cart_in11')

merge_both = users.merge(diff_carts_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_brand_cart_in11'] = np.where(merge_both['diff_brands_carted'] < 1, merge_both['diff_brands_carted'], merge_both['cart_in11']/merge_both['diff_brands_carted'])
age_gender = age_gender.merge(merge_both[['user_id', 'ratio_brand_cart_in11']], how="left", on=["user_id"]).fillna(0)

# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_brand_cart_in11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
del merge_both, diff_carts_11, carts_in11

#-------------------------------- F:32 --------------------------------------------

#users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_brands_bought'])
print('32')
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_brands_bought'] = age_gender['diff_brands_bought']

buys_in11 = df_logs[(df_logs.action_type == 2) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_buy_11 = buys_in11.drop_duplicates(subset=['brand_id'])
diff_buy_11 = diff_buy_11.groupby('user_id').size().reset_index(name='buy_in11')

merge_both = users.merge(diff_buy_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_brand_buy_in11'] = np.where(merge_both['diff_brands_bought'] < 1, merge_both['diff_brands_bought'], merge_both['buy_in11']/merge_both['diff_brands_bought'])
age_gender = age_gender.merge(merge_both[['user_id', 'ratio_brand_buy_in11']], how="left", on=["user_id"]).fillna(0)

# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_brand_buy_in11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
del merge_both, diff_buy_11, buys_in11
#
# #------------------------------- F:33 -----------------------------------------------
#
# users =  pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', skipinitialspace=True, usecols=['user_id', 'diff_brands_favorited'])
print('33')
users = pd.DataFrame()
users['user_id'] = age_gender['user_id']
users['diff_brands_favorited'] = age_gender['diff_brands_favorited']

favs_in11 = df_logs[(df_logs.action_type == 3) & ((df_logs.time_stamp > 1031) & (df_logs.time_stamp < 1131))].sort_values(['user_id'],ascending=True)

diff_fav_11 = favs_in11.drop_duplicates(subset=['brand_id'])
diff_fav_11 = diff_fav_11.groupby('user_id').size().reset_index(name='fav_in11')

merge_both = users.merge(diff_fav_11, how="left", on=['user_id']).fillna(0)
merge_both['ratio_brand_fav_in11'] = np.where(merge_both['diff_brands_favorited'] < 1, merge_both['diff_brands_favorited'], merge_both['fav_in11']/merge_both['diff_brands_favorited'])
age_gender = age_gender.merge(merge_both[['user_id', 'ratio_brand_fav_in11']], how="left", on=["user_id"]).fillna(0)

# df = pd.read_csv('/home/hassan/Desktop/SalesAI2/user-features.csv')
# new_column = pd.DataFrame({'ratio_brand_fav_in11': merge_both['ratio']})
# df = df.merge(new_column, left_index = True, right_index = True)
# df.to_csv('/home/hassan/Desktop/SalesAI2/user-features.csv', index=False)
# del merge_both, diff_fav_11, favs_in11

#--------------------------Calculate Time for User Ratio Features-------------------------------------------------------
stop = timeit.default_timer()
print('Total Time to calculate user-ratio-features: ' , (stop - start) , ' Seconds.')
print('Total Time to calculate user-ratio-features: ' ,((stop - start)/60) , ' Minutes.')
#-------------Write all features to file--------------------------------------------------------------------------------
age_gender.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/user-related-features.csv', index=False)

