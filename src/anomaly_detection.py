import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


class AnomalyDetection:

    def __init__(self, weather_df, generation_df):
        self.weather_df = weather_df
        self.generation_df = generation_df

        # preprocessing
        self.weather_df['DATE_TIME'] = pd.to_datetime(
            self.weather_df['DATE_TIME'])
        self.generation_df['DATE_TIME'] = pd.to_datetime(
            self.generation_df['DATE_TIME'])

        self.weather_df['DAY'] = pd.DatetimeIndex(
            self.weather_df['DATE_TIME']).dayofyear
        self.weather_df['TIME'] = self.weather_df.DATE_TIME.dt.hour * \
            60 + self.weather_df.DATE_TIME.dt.minute
        self.weather_df['Time'] = self.weather_df.DATE_TIME.dt.time
        self.weather_df['HOUR'] = self.weather_df.DATE_TIME.dt.hour

        self.merged_df = pd.merge(self.generation_df, self.weather_df, how='inner', on=[
                                  'DATE_TIME'], suffixes=('', '_y'))


    def find_cloudiness_v1(self, col_name):
        # Removing outliers from the max curve
        clean_data = self.weather_df.copy()
        for i, j in clean_data.groupby(clean_data.HOUR):
            outlier_condition1 = (j[col_name] != 0) & (
                j[col_name] >= (j[col_name].mean() + 3*j[col_name].std()))
            outlier_condition2 = (j[col_name] != 0) & (
                j[col_name] < (j[col_name].mean() - 3*j[col_name].std()))
            clean_data.drop(clean_data.loc[(clean_data['DAY'].isin(j.loc[outlier_condition1 | outlier_condition2].DAY)) &
                                           (clean_data['HOUR'] == i)].index, inplace=True)
        agg_h_irr_clean = clean_data.groupby(
            clean_data.TIME).agg({col_name: 'max'})
        weather_maxirr = pd.merge(
            self.weather_df, agg_h_irr_clean, how='inner', on='TIME', suffixes=('', '_max'))
        weather_maxirr['offset_from_max'] = (
            weather_maxirr[col_name+'_max']-weather_maxirr[col_name])**2
        C_day_list = weather_maxirr.groupby('DAY').sum()['offset_from_max']

        output = pd.DataFrame(C_day_list)
        output.columns = ['cloudiness']
        output.reset_index(inplace=True)
        return output

    def find_cloudiness_v2(self, col_name, neighbour_number=10, outlier_limit=10):
        # Removing outliers from the max curve
        clean_data = self.weather_df.copy()
        X = clean_data[['TIME', col_name]]
        nbrs = NearestNeighbors(n_neighbors=neighbour_number).fit(X)
        distances, indices = nbrs.kneighbors(X)
        distance = pd.DataFrame(distances).loc[:, 1:].mean(axis=1)
        outlier_condition = distance > distance.mean()+outlier_limit*distance.std()
        clean_data.drop(distance[outlier_condition].index, inplace=True)
        agg_h_irr_clean = clean_data.groupby(
            clean_data.TIME).agg({col_name: 'max'})
        weather_maxirr = pd.merge(
            self.weather_df, agg_h_irr_clean, how='inner', on='TIME', suffixes=('', '_max'))
        weather_maxirr['offset_from_max'] = (
            weather_maxirr[col_name+'_max']-weather_maxirr[col_name])**2
        C_day_list = weather_maxirr.groupby('DAY').sum()['offset_from_max']
        output = pd.DataFrame(C_day_list)
        output.columns = ['cloudiness']
        output.reset_index(inplace=True)
        return output

    def get_outliers_in_time(self, column, groupby_column, output_column_name=False, outlier_limit=3):
        if output_column_name == False:
            output_column_name = column+"_outliers"
        self.merged_df[output_column_name] = 0
        for i, group in self.merged_df.groupby(groupby_column):
            outlier_condition1 = group[column] > (group[group[column] > 0][column].mean(
            ) + outlier_limit*group[group[column] > 0][column].std())
            outlier_condition2 = group[column] < (group[group[column] > 0][column].mean(
            ) - outlier_limit*group[group[column] > 0][column].std())
            self.merged_df.loc[group[(outlier_condition1 | outlier_condition2) & (
                group[group[column] > 0][column].std() > 0)].index, output_column_name] = 1
        return self.merged_df

    def get_outliers_for_fit(self, x_column, y_column, groupby_column, output_column_name=False, window=False, outlier_limit=3):
        if output_column_name == False:
            output_column_name = x_column+"_outliers"  # ?
        self.merged_df[output_column_name] = 0
        self.merged_df['gross_efficiency'] = self.merged_df[y_column] / \
            self.merged_df[x_column]
        for i, group in self.merged_df.groupby(groupby_column):
            resorted = group.reset_index().set_index('DATE_TIME').copy()
            outlier_condition1 = resorted[resorted[y_column] > 0]['gross_efficiency'] > (resorted[resorted[y_column] > 0]['gross_efficiency'].rolling(
                window=window).mean() + (outlier_limit+1)*resorted[resorted[y_column] > 0]['gross_efficiency'].rolling(window=window).std())
            outlier_condition2 = resorted[resorted[y_column] > 0]['gross_efficiency'] < (resorted[resorted[y_column] > 0]['gross_efficiency'].rolling(
                window=window).mean() - outlier_limit*resorted[resorted[y_column] > 0]['gross_efficiency'].rolling(window=window).std())
            outlier_condition3 = resorted[resorted[y_column] > 0][x_column] > 0.1 * \
                resorted[resorted[y_column] > 0][x_column].rolling(
                    window=window).max()
            self.merged_df.loc[resorted[resorted[y_column] > 0][(
                outlier_condition1 | outlier_condition2) & outlier_condition3]['index'], output_column_name] = 1
            self.merged_df.loc[resorted[(resorted[y_column] == 0) & (
                resorted[x_column] > 0.1*resorted[x_column].rolling(window=window).max())]['index'], output_column_name] = 1
        return self.merged_df
    # function returns the data with the heighest and lowest residuals

    def get_outliers_by_residual(self, x_column, y_column, output_column_name="residual_outliers", anomaly_limit=5):
        self.merged_df[output_column_name] = 0
        for i in self.merged_df.SOURCE_KEY.unique():
            inv_data = self.merged_df[self.merged_df.SOURCE_KEY == i]
            for _, day in inv_data.groupby(inv_data.DAY):
                X = self.merged_df[x_column].values.reshape(-1, 1)
                y = self.merged_df[y_column].values.reshape(-1, 1)
                regressor = LinearRegression(fit_intercept=False)
                regressor.fit(X, y)
                m = regressor.coef_[0]
                # or we can take the absolute value
                residual = (day[y_column]-m*day[x_column])**2
                mean_res = residual.mean()
                stand_dev = residual.std()
                self.merged_df.loc[day[(residual > mean_res+anomaly_limit*stand_dev) | (
                    residual < mean_res-anomaly_limit*stand_dev)].index, output_column_name] = 1
        return self.merged_df

    # Power conversion efficiency calculation (Irradiation to DC or DC to AC)
    @staticmethod
    def get_conversion_coefficients(data, x_column, y_column, output_column_name="conversion_coefficient"):
        output_df = pd.DataFrame(
            columns=['SOURCE_KEY', 'DAY', output_column_name])
        for i in data.SOURCE_KEY.unique():
            inv_data = data[data.SOURCE_KEY == i]
            for a, day in inv_data.groupby(inv_data.DAY):
                X = day[x_column].values.reshape(-1, 1)
                y = day[y_column].values.reshape(-1, 1)
                regressor = LinearRegression(fit_intercept=False)
                regressor.fit(X, y)
                m = regressor.coef_[0][0]
                output_df = output_df.append(pd.DataFrame(
                    {'SOURCE_KEY': [i], 'DAY': [a], output_column_name: [m]}), ignore_index=True)
        return output_df

    @staticmethod
    def negative_trend_by_days(data, column, cloudiness, output_column_name, window=10, limit=-0.75):
        """
        Given a dataframe of inverter daily efficiencies calculates negative trends over a window of days. 
        Returns the dataframe with a new column, with 1 where the negative trend is below a certain limit.
        """
        data = data.merge(cloudiness, on='DAY')
        data['efficiency_trend'] = 0
        data['weather_trend'] = 0
        data[output_column_name] = 0
        for inv in data.SOURCE_KEY.unique():
            data.loc[data.SOURCE_KEY == inv, 'efficiency_trend'] = (data[data.SOURCE_KEY == inv]['DAY']/data[data.SOURCE_KEY == inv]['DAY'].max(
            )).rolling(window=window).corr(data[data.SOURCE_KEY == inv][column]/data[data.SOURCE_KEY == inv][column].max())
            #data.loc[data.SOURCE_KEY==inv, 'weather_trend']=(data[data.SOURCE_KEY==inv][column]/data[data.SOURCE_KEY==inv][column].max()).rolling(window=window).corr(data[data.SOURCE_KEY==inv]['cloudiness']/data[data.SOURCE_KEY==inv]['cloudiness'].max())
        data.loc[(data['efficiency_trend'] <= limit), output_column_name] = 1
        ####
        return data.drop(columns=['efficiency_trend', 'weather_trend', 'cloudiness'])

    @staticmethod
    def efficiency_drop_by_day(data, column, outlier_coeff, drop_column_name, jump_column_name, window=20):
        """
        Given a dataframe of inverter daily efficiencies calculates efficiency jumps and falls relative to inverter history.
        """
        if drop_column_name:
            data[drop_column_name] = 0
        if jump_column_name:
            data[jump_column_name] = 0
        data['efficiency_lower_limit'] = 0
        data['efficiency_higher_limit'] = 0
        if window:
            for inv in data.SOURCE_KEY.unique():
                data.loc[data.SOURCE_KEY == inv, 'efficiency_lower_limit'] = data[data.SOURCE_KEY == inv][column].rolling(
                    window=window).median()-outlier_coeff*data[data.SOURCE_KEY == inv][column].rolling(window=window).std()
                data.loc[data.SOURCE_KEY == inv, 'efficiency_higher_limit'] = data[data.SOURCE_KEY == inv][column].rolling(
                    window=window).median()+outlier_coeff*data[data.SOURCE_KEY == inv][column].rolling(window=window).std()
        else:
            for inv in data.SOURCE_KEY.unique():
                data.loc[data.SOURCE_KEY == inv, 'efficiency_lower_limit'] = data[data.SOURCE_KEY ==
                                                                                  inv][column].median()-outlier_coeff*data[data.SOURCE_KEY == inv][column].std()
                data.loc[data.SOURCE_KEY == inv, 'efficiency_higher_limit'] = data[data.SOURCE_KEY ==
                                                                                   inv][column].median()+outlier_coeff*data[data.SOURCE_KEY == inv][column].std()
        if drop_column_name:
            data.loc[data[column] < data['efficiency_lower_limit'],
                     drop_column_name] = 1
        if jump_column_name:
            data.loc[data[column] > data['efficiency_higher_limit'],
                     jump_column_name] = 1
        return data.drop(columns=['efficiency_lower_limit', 'efficiency_higher_limit'])

    @staticmethod
    def get_inefficient_inverters_day(data, column, output_column_name, anomaly_limit=2):
        data[output_column_name] = 0
        for _, d in data.groupby('DAY'):
            col_mean = d[column].mean()
            col_std = d[column].std()
            data.loc[d[(d[column] < col_mean - anomaly_limit *
                        col_std)].index, output_column_name] = 1
        return data

    @staticmethod
    def get_inefficient_inverters_window(data, column, output_column_name, window=7):
        data[output_column_name] = 0
        for inv in data['SOURCE_KEY'].unique():
            data.loc[data['SOURCE_KEY'] == inv, output_column_name] = data[data['SOURCE_KEY']
                                                                           == inv][column].rolling(window=window).min()
        return data

    def start(self):
        cloudiness_v2 = self.find_cloudiness_v2('IRRADIATION')
        self.merged_df = self.get_outliers_for_fit(
            'IRRADIATION', 'DC_POWER', 'SOURCE_KEY', output_column_name="alarm_DC_conversion_outlier", window='7d', outlier_limit=4)
        inv0 = self.merged_df[self.merged_df.SOURCE_KEY == self.merged_df.SOURCE_KEY.unique()[
            0]]
        self.merged_df = self.get_outliers_by_residual(
            'DC_POWER', 'AC_POWER', output_column_name="alarm_AC_conversion_outlier", anomaly_limit=5)

        # Power conversion efficiency calculation (Irradiation to DC or DC to AC)
        # DC generation
        daily_data = self.get_conversion_coefficients(self.merged_df[(self.merged_df['MODULE_TEMPERATURE'] <= 50) & (
            self.merged_df["alarm_DC_conversion_outlier"] == 0)], 'IRRADIATION', 'DC_POWER', output_column_name="DC_efficiency_unnorm")
        daily_data['DC_efficiency'] = daily_data['DC_efficiency_unnorm'] / \
            daily_data['DC_efficiency_unnorm'].max()
    
        daily_data_AC = self.get_conversion_coefficients(self.merged_df[(
            self.merged_df["alarm_AC_conversion_outlier"] == 0)], 'DC_POWER', 'AC_POWER', output_column_name="AC_efficiency_unnorm")
        daily_data_AC['AC_efficiency'] = daily_data_AC['AC_efficiency_unnorm'] / \
            daily_data_AC['AC_efficiency_unnorm'].max()
        daily_data.head()
        daily_data = daily_data.merge(daily_data_AC, on=['SOURCE_KEY', 'DAY'])
        
        # Detection of efficiency drops, trends, or inefficient behaviour compared to other inverters
        daily_data = self.negative_trend_by_days(
            daily_data, 'DC_efficiency', cloudiness_v2, output_column_name='alarm_negative_trend', window=10, limit=-0.8)
        
        daily_data = self.negative_trend_by_days(
            daily_data, 'AC_efficiency', cloudiness_v2, output_column_name='alarm_AC_negative_trend', window=10, limit=-0.6)

        daily_data[daily_data['alarm_AC_negative_trend'] == 1]
        

        daily_data = self.efficiency_drop_by_day(
            daily_data, 'DC_efficiency', 2, 'alarm_DC_conversion_fall', 'alarm_DC_conversion_jump', window=7)
        inverters = daily_data[daily_data['alarm_DC_conversion_fall']
                               == 1]['SOURCE_KEY'].unique()
        data = daily_data[daily_data['SOURCE_KEY'].isin(inverters)]
        

        daily_data = self.efficiency_drop_by_day(
            daily_data, 'AC_efficiency', 2, 'alarm_AC_conversion_fall', 'alarm_AC_conversion_jump', window=7)
        inverters = daily_data[daily_data['alarm_AC_conversion_fall']
                               == 1]['SOURCE_KEY'].unique()
        data = daily_data[daily_data['SOURCE_KEY'].isin(inverters)]

        daily_data = self.get_inefficient_inverters_day(
            daily_data, 'DC_efficiency', 'alarm_inefficient_inverter_day')
        data = daily_data
        
        daily_data = self.get_inefficient_inverters_day(
            daily_data, 'AC_efficiency', 'alarm_AC_inefficient_inverter_day')
        data = daily_data
        
        return self.merged_df, daily_data

def main(weather_path, generation_path, output_path1, output_path2):
    if not os.path.isfile(weather_path):
        raise Exception(
            f"{weather_path} is not a dataset. Please make sure that the specifced path is correct.")
    if not os.path.isfile(generation_path):
        raise Exception(
            f"{generation_path} is not a dataset. Please make sure that the specifced path is correct.")
    weather_df = pd.read_csv(weather_path)
    generation_df = pd.read_csv(generation_path)

    anomaly_detection = AnomalyDetection(weather_df, generation_df)
    output_data_full, daily_output=anomaly_detection.start()

    output_data_full.to_csv(
            '../reports/alarm_inefficient_inverter_day.csv', index=False)
    daily_output.to_csv(
            '../reports/alarm_inefficient_inverter_day.csv', index=False)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("Please specify the weather and generation datasets")
    main(weather_path=sys.argv[1], generation_path=sys.argv[2])
