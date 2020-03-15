
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
import lstmModel

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

#train_model(model, X_train, y_train, X_test, y_test)
def train_model(model, train_data, train_labels, test_data=None, test_labels=None):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 60

  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred = model(train_data)

    loss = loss_fn(y_pred.float(), train_labels)

    if test_data is not None:
      with torch.no_grad():
        y_test_pred = model(test_data)
        test_loss = loss_fn(y_test_pred.float(), test_labels)
      test_hist[t] = test_loss.item()

      if t % 10 == 0:  
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 10 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), train_hist, test_hist

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
    
    #######################################################################

    test_data_size = int(0.3*len(daily_cases))
    train_data = daily_cases[:-test_data_size]
    test_data = daily_cases[-test_data_size:]
    #train_data.shape
    
    scaler = MinMaxScaler() #values between 0-1
    scaler = scaler.fit(np.expand_dims(train_data, axis=1))
    train_data = scaler.transform(np.expand_dims(train_data, axis=1))
    test_data = scaler.transform(np.expand_dims(test_data, axis=1))

    def create_sequences(data, seq_length):
        xs = []
        ys = []

        for i in range(len(data)-seq_length-1):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    seq_length = 10
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    model = lstmModel.CoronaVirusPredictor(n_features=1, n_hidden=512, seq_len=seq_length, n_layers=2)
    model, train_hist, test_hist = train_model(model, X_train, y_train, X_test, y_test)

    plt.plot(train_hist, label="Training loss")
    plt.plot(test_hist, label="Test loss")
    plt.ylim((0, 5))
    plt.legend()
    plt.show()


    with torch.no_grad():
        test_seq = X_test[:1]
        preds = []
        for _ in range(len(X_test)):
            y_test_pred = model(test_seq)
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
            new_seq = test_seq.numpy().flatten()
            new_seq = np.append(new_seq, [pred])
            new_seq = new_seq[1:]
            test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
    
    true_cases = scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()
    predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()

    plt.plot(daily_cases.index[:len(train_data)], scaler.inverse_transform(train_data).flatten(),label='Historical Daily Cases')
    plt.plot(daily_cases.index[len(train_data):len(train_data) + len(true_cases)], true_cases,label='Real Daily Cases')
    plt.plot(daily_cases.index[len(train_data):len(train_data) + len(true_cases)], predicted_cases, label='Predicted Daily Cases')
    plt.show()






if __name__ == '__main__':
	main()