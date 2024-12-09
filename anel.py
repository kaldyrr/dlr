import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

file_path = 'VadimAnel.xlsx' 
data = pd.read_excel(file_path)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data = data.set_index('Date')
full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
data = data.reindex(full_date_range).ffill().reset_index()
data.columns = ['Date', 'Rate']
scaler = MinMaxScaler(feature_range=(0, 1))
data['Rate'] = scaler.fit_transform(data[['Rate']])
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
data_values = data['Rate'].values.reshape(-1, 1)
time_step = 60  
X, y = create_dataset(data_values, time_step)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=50, return_sequences=False)) 
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
# Оценка модели
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Train Loss: {train_loss}')
print(f'Test Loss: {test_loss}')
# Предсказания и вывод метрик
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_pred = scaler.inverse_transform(train_pred)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_pred = scaler.inverse_transform(test_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
mae = np.mean(np.abs(test_pred - y_test))
rmse = np.sqrt(np.mean((test_pred - y_test)**2))
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
# Визуализация результатов (график обучающих и тестовых данных)
plt.figure(figsize=(12, 6))
plt.plot(data['Date'][:len(y_train)], y_train, color='blue', label='Actual Train Data')
plt.plot(data['Date'][len(y_train):len(y_train) + len(y_test)], y_test, color='red', label='Actual Test Data')
plt.plot(data['Date'][len(y_train):len(y_train) + len(test_pred)], test_pred, color='green', label='Predicted Test Data')
plt.legend()
plt.title('Currency Rate Prediction using LSTM (Training & Test Data)')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.show()
last_sequence = data_values[-time_step:] 
last_sequence = np.array(last_sequence).reshape(1, -1, 1)
predictions = []
for i in range(6):  
    temp_predictions = []
    for j in range(30):  
        pred = model.predict(last_sequence)[0][0]
        temp_predictions.append(pred)
        last_sequence = np.roll(last_sequence, -1, axis=1)  
        last_sequence[0, -1, 0] = pred  
    predictions.extend(temp_predictions)
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=180, freq='D')
plt.figure(figsize=(12, 6))
plt.plot(forecast_dates, predictions, color='orange', label='Predicted 180 Days')
plt.legend()
plt.title('Currency Rate Forecast for Next 180 Days')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.show()
