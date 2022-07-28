#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

plant_gen = pd.read_csv('../data/Plant_1_Generation_Data.csv')
plant_weat = pd.read_csv('../data/Plant_1_Weather_Sensor_Data.csv')

# Convert DATR_TIME from string to datetime
plant_gen['DATE_TIME'] = pd.to_datetime(plant_gen['DATE_TIME'])
plant_weat['DATE_TIME'] = pd.to_datetime(plant_weat['DATE_TIME'])


len(plant_gen) == len(plant_gen.dropna())

# This implies that there are no nas in the dataset

len(plant_weat) == len(plant_weat.dropna())

# This implies that there are no nas in the dataset


plant_gen['dt'] = plant_gen['DATE_TIME']
plant_weat['dt'] = plant_weat['DATE_TIME']
plant_gen = plant_gen.set_index('DATE_TIME')
plant_weat = plant_weat.set_index('DATE_TIME')


may_15 = plant_gen['2020-05-15 00:00:00':'2020-05-16 00:00:00']

plt.figure(figsize=(20, 10))
for key in may_15['SOURCE_KEY'].unique():
    plt.plot(may_15[may_15['SOURCE_KEY'] == key].index, may_15[may_15['SOURCE_KEY'] == key]['AC_POWER'], color='black')

for _, gp in may_15.groupby(by=may_15.index):
    plt.plot(_, gp['AC_POWER'].mean(), marker='o', color='red')
    plt.plot(_, gp['AC_POWER'].mean() + 3 * gp['AC_POWER'].std(), marker='o', color='blue')
    plt.plot(_, gp['AC_POWER'].mean() - 3 * gp['AC_POWER'].std(), marker='o', color='blue')
    
plt.show()

may_15 = plant_gen['2020-05-19 00:00:00':'2020-05-20 00:00:00']
plt.figure(figsize=(20, 10))
for key in may_15['SOURCE_KEY'].unique():
    plt.plot(may_15[may_15['SOURCE_KEY'] == key].index, may_15[may_15['SOURCE_KEY'] == key]['AC_POWER'], color='black')

for _, gp in may_15.groupby(by=may_15.index):
    plt.plot(_, gp['AC_POWER'].mean(), marker='o', color='red')
    plt.plot(_, gp['AC_POWER'].mean() + 3 * gp['AC_POWER'].std(), marker='o', color='blue')
    plt.plot(_, gp['AC_POWER'].mean() - 3 * gp['AC_POWER'].std(), marker='o', color='blue')
    
plt.show()


may_15 = plant_gen['2020-06-17 00:00:00':'2020-06-18 00:00:00']
plt.figure(figsize=(20, 10))
for key in may_15['SOURCE_KEY'].unique():
    plt.plot(may_15[may_15['SOURCE_KEY'] == key].index, may_15[may_15['SOURCE_KEY'] == key]['AC_POWER'], color='black')

for _, gp in may_15.groupby(by=may_15.index):
    plt.plot(_, gp['AC_POWER'].mean(), marker='o', color='red')
    plt.plot(_, gp['AC_POWER'].mean() + 3 * gp['AC_POWER'].std(), marker='o', color='blue')
    plt.plot(_, gp['AC_POWER'].mean() - 3 * gp['AC_POWER'].std(), marker='o', color='blue')
    
plt.show()


for day in sorted(plant_weat['dt'].dt.dayofyear.unique()):
    print(str(datetime(2020, 1, 1) + timedelta(int(day) - 1)))
print('-----------------------------')
for day in sorted(plant_gen['dt'].dt.dayofyear.unique()):
    print(str(datetime(2020, 1, 1) + timedelta(int(day) - 1)))

merged_df = pd.merge(plant_gen, plant_weat, how='inner', on='DATE_TIME')


plt.figure(figsize=(20, 10))
plt.scatter(merged_df['MODULE_TEMPERATURE'], merged_df['AC_POWER'])



plt.figure(figsize=(20, 10))
plt.scatter(merged_df['AMBIENT_TEMPERATURE'], merged_df['AC_POWER'])


plt.figure(figsize=(20, 10))
plt.scatter(plant_weat['AMBIENT_TEMPERATURE'], plant_weat['MODULE_TEMPERATURE'])


plt.figure(figsize=(20, 10))
plt.scatter(plant_weat['IRRADIATION'], plant_weat['MODULE_TEMPERATURE'])



INV_2 = plant_gen[plant_gen['SOURCE_KEY'] == plant_gen.SOURCE_KEY.unique()[2]]



for _, gp in INV_2.groupby(INV_2['dt'].dt.dayofyear):
    plt.figure(figsize=(15, 10))
    plt.plot_date(gp['dt'], gp['AC_POWER'])
plt.show()


INV_2 = INV_2.resample('D').agg({
    'AC_POWER' : 'sum'
})


sum(INV_2['AC_POWER'] == 0)/len(INV_2)



plt.figure(figsize=(15, 10))
plt.plot_date(INV_2.index, INV_2['AC_POWER'])

