from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from model import get_model

# val_set_per = 10
# test_set_per = 10

def scale(df, inverse=False):
    scaler = preprocessing.MinMaxScaler()
    if inverse:
        df = scaler.fit_transform(tf.reshape(df, (-1,1)))
        return df
    else:
        df = scaler.inverse_transform(tf.reshape(df, (-1,1)))
        return df


def get_prediction(df, window_size=60):
    df_eqix = df.copy()
    df_eqix.dropna(inplace=True)
    df_eqix_open = df_eqix['open']
    df_eqix_close = df_eqix['close']
    df_eqix_high = df_eqix['high']
    df_eqix_low = df_eqix['low']
    df_eqix_open_norm = scale(df_eqix_open.copy())
    df_eqix_close_norm = scale(df_eqix_close.copy())
    df_eqix_high_norm = scale(df_eqix_high.copy())
    df_eqix_low_norm = scale(df_eqix_low.copy())
    # df_eqix_norm = scale(df_eqix_norm)
    # x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(df_eqix_norm, window_size=window_size)
    model = get_model(window_size)
    pred_open = model.predict(df_eqix_open_norm)
    pred_close = model.predict(df_eqix_close_norm)
    pred_high = model.predict(df_eqix_high_norm)
    pred_low = model.predict(df_eqix_low_norm)
    return [scale(pred_open, True), scale(pred_close, True), scale(pred_high, True), scale(pred_low,True)]
    
