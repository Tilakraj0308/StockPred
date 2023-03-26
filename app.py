from flask import Flask, render_template, jsonify
import yfinance as yf
import json

# apple = Share('YHOO')
# s = apple.data_set()
# print(type(s))

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    f = open("tickers.json")
    data = json.load(f)
    f.close()
    print(data)
    return render_template("index.html",  tickers=data)

@app.route('/<symbol>', methods=['GET'])
def symbol(symbol):
   
    return render_template("index.html")


@app.route('/data/<symbol>', methods=['GET'])
def data(symbol="MSFT"):
    stock = yf.Ticker(str(symbol))
    history = stock.history(period="1mo")
    return jsonify(history.to_json(orient='table'))