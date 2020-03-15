
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
import wget

import Exporter

def init():
    #matplotlib inline
    #config InlineBackend.figure_format='retina'

    sns.set(style='whitegrid', palette='muted', font_scale=1.2)

    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

    rcParams['figure.figsize'] = 14, 10
    register_matplotlib_converters()

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

def getData():
    try:
        f = open("time_series_19-covid-Confirmed.csv")
        f.close()
    except IOError:
        wget.download("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")

    #ourfile = open(filename)
    #print(ourfile.read())

def analyzeData():
    chart = pd.read_csv("time_series_19-covid-Confirmed.csv")
    #print(chart.head())
    
    chart = chart.iloc[:, 4:]

    daily_cases = chart.sum(axis=0)
    daily_cases.index = pd.to_datetime(daily_cases.index)
    plt.plot(daily_cases)
    plt.title("Cumulative daily cases")
    plt.show()

    daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
    plt.plot(daily_cases)
    plt.title("daily cases")
    plt.show()

    return daily_cases


def main():
    
    getData()
    
    daily_cases = analyzeData()
    
    try:
        f = open("output_got.csv","r+", encoding="utf8")
        f.close()
    except IOError:
        Exporter.getTweets("output_got.csv","crono virus","2020-02-12","2020-02-15")

    print(open("output_got.csv","r+", encoding="utf8").read())
    print("------------------------------------------------------------------------------------")
    try:
        f = open("output_got2.csv","r+", encoding="utf8")
        f.close()
    except IOError:
        Exporter.getTweets("output_got2.csv","crono virus","2020-03-12","2020-03-15")

    print(open("output_got2.csv","r+", encoding="utf8").read())
        
    
    test_data_size = int(0.3*len(daily_cases))
    train_data = daily_cases[:-test_data_size]
    test_data = daily_cases[-test_data_size:]
    #train_data.shape




if __name__ == '__main__':
	main()