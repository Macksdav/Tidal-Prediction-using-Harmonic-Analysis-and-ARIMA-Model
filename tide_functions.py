import pandas as pd 
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from ttide import t_tide, t_utils
from matplotlib.dates import date2num
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Creating a function that would put the data in its correct data type
# Plotting the Data for the different study area
def analysis_plotter(data,plot_title):
    fig = plt.figure(figsize=(17, 6))
    ax = fig.add_subplot(111)
    sns.lineplot(data=data, x= data.iloc[:,0], y= data.iloc[:,1])
    ax.set_ylabel('Surface elevation (m)')
    plt.title(plot_title)

    return fig.show()


def convert_datatype(data):
    converted_date = []
    for i in data.iloc[:,0]:
        date_string = f"{i}"
        date_format = "%m/%d/%Y %H:%M"
        conversion = datetime.strptime(date_string, date_format)
        converted_date.append(conversion)

    data.iloc[:,0] = pd.Series(converted_date)
    return data


# Check for stationarity
def stationarity_test(data):
    result = adfuller(data.iloc[:,1])
    return f'ADF Statistic: {result[0]}, p-value: {result[1]}'


def acf_pacf_plot(data):
    plot_acf(data.iloc[:,1], title='ACF Plot')
    plot_pacf(data.iloc[:,1], title='PACF Plot')
    return plt.show()


def moving_average(data):
    ma = data.iloc[:,1].rolling(50).mean().plot(ylim=[0.5, 1.5], title='Moving Average')
    return ma


def train_test(data):
    x_train = data[:round(len(data)*0.65)]
    x_test = data[round(len(data)*0.65):]

    return (x_train, x_test)


def model_training(x_train,x_test):
    arima_model = ARIMA(x_train.iloc[:,1], order= (50,0,1))
    model = arima_model.fit()
    forecast = model.forecast(steps=500)
    plt.figure(figsize=(14, 7))
    plt.plot(x_train.iloc[:,1], label='Observed Train')  # Plot the training data
    plt.plot(x_test.iloc[:,1], label='Observed Test', color='green')  # Plot the test data
    plt.plot(forecast, label='Forecast', linestyle='--', color='orange')  # Plot the forecasted data
    plt.title('ARIMA Model Forecast vs Actual')  # Set the plot title
    plt.legend()  # Add a legend
    

    
    print(model.summary())
    return plt.show()