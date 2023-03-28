from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from model import get_model
import yfinance as yf


# val_set_per = 10
# test_set_per = 10



# def scale(df, inverse=False):
#     scaler = preprocessing.MinMaxScaler()
#     if not inverse:
#         df = scaler.fit_transform(tf.reshape(df, (-1,1)))
#         return df
#     else:
#         df = scaler.inverse_transform(df)
#         return df


# def get_prediction(df, model):
#     # Scale
#     scaler = preprocessing.MinMaxScaler()
#     scaler.fit(tf.reshape(df, (-1,1)))
#     df_scaled = scaler.transform(tf.reshape(df, (-1, 1)))
#     df_scaled = tf.expand_dims(df_scaled, axis=0)
#     pred = model.predict(df_scaled)
#     return scaler.inverse_transform(tf.reshape(pred, (-1, 1)))


# for continous prediction using sliding window of size 'n'
def get_prediction_cont(df, model, upto_predict, size=30):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(tf.reshape(df, (-1,1)))
    df_scaled = scaler.transform(tf.reshape(df, (-1, 1)))
    df_scaled = df_scaled[1:]
    pred = []
    # print(df_scaled.shape)
    # print(df_scaled[0].shape)
    for i in range(upto_predict):
        df2 = df_scaled[1-size:]
        # shape(None, n, 1) --> Shape(1, n, 1)
        df2 = tf.expand_dims(df2, axis=0)
        # predicted_scaled has (1,1) output but should be (1,) for adding in the df_scaled
        predicted_scaled = model.predict(df2)
        # Appending into df_scaled but it reduces its shape from (29, 1) to (30,) so we need to expand dims
        df_scaled = tf.expand_dims(np.append(df_scaled, tf.squeeze(predicted_scaled, axis = -1)), axis=1)
        predicted = scaler.inverse_transform(predicted_scaled)
        pred.append(predicted)
    return np.array(pred)

    
def get_full_prediction(df, upto_predict, window_size=30):
    df_eqix = df.copy()
    df_eqix.dropna(inplace=True)
    pred = []
    components = [df_eqix['Open'], df_eqix['Close'], df_eqix['High'], df_eqix['Low']]
    model = get_model(window_size)
    for c in components:
        # c = c[1:]
        # print(c.shape)
        pred.append(get_prediction_cont(c, model, upto_predict, window_size))
    return pred


