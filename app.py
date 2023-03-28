from flask import Flask, jsonify
import yfinance as yf
from util import get_full_prediction

app = Flask(__name__)


@app.route('/stocks/<symbol>/predict', methods=['GET', 'POST'])
def predict(symbol, upto_predict=30):
    stock = yf.Ticker(str(symbol))
    history = stock.history(period="30d")
    pred = get_full_prediction(history, upto_predict, window_size=30)
    # print(history['Open'])
    return str(len(pred[0]))