import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import yfinance as yf
import numpy as np

app = dash.Dash()
server = app.server

def load_trained_model(crypto):
    model=load_model(f"./models/{crypto}.h5")
    scaler = load(f"./models/{crypto}.joblib")
    return model, scaler

def load_data(crypto):
    df = yf.download(tickers=crypto, period='1mo', interval='5m')
    return df[-2000:][['Close']]

def prepare_data(crypto):
    df = load_data(crypto)
    dataset = df.values
    model, scaler = load_trained_model(crypto)

    X_test = []
    X_test = []
    scaled_data = scaler.transform(dataset)
    for i in range(60, len(scaled_data)):
        X_test.append(scaled_data[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction.flatten()

    return df, prediction

app.layout = html.Div([html.H1("LSTM Stock price prediction", style={"textAlign": "center"}),
    html.Div([
        dcc.Dropdown(id='my-dropdown',
                        options=[{'label': 'BTC-USD', 'value': 'BTC-USD'},
                                {'label': 'ETH-USD','value': 'ETH-USD'}, 
                                {'label': 'ADA-USD', 'value': 'ADA-USD'}], 
                        value='BTC-USD',
                        style={"display": "block", "margin-left": "auto", 
                            "margin-right": "auto", "width": "60%"}),
        dcc.Graph(
            id="graph",
        )
    ])
])

@app.callback(Output('graph', 'figure'), Input('my-dropdown', 'value'))
def update_graph(selected_dropdown):
    df, prediction = prepare_data(selected_dropdown)
    trace1 = []
    trace2 = []
    trace1.append(
        go.Scatter(x=df.index,
                    y=df['Close'],
                    mode='lines', opacity=0.7, 
                    name=f'Actual price',textposition='bottom center'))
    trace2.append(
        go.Scatter(x=df.index[60:],
                    y=prediction,
                    mode='lines', opacity=0.6,
                    name=f'Predicted price', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual vs Predicted price for {selected_dropdown}",
            xaxis={"title":"Time"},
            yaxis={"title":"Price (USD)"})}
    return figure



if __name__=='__main__':
    app.run_server(debug=True)