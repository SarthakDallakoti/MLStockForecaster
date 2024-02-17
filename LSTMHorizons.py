import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Define functions
def calculate_technical_indicators(data):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.abs(np.where(delta < 0, delta, 0))
    
    gain_series = pd.Series(gain, index=data.index)
    loss_series = pd.Series(loss, index=data.index)
    
    avg_gain = gain_series.rolling(window=14).mean()
    avg_loss = loss_series.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Fetch data
ticker_symbol = '^GSPC'
start_date = '2000-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
ticker_data = yf.Ticker(ticker_symbol)
df = ticker_data.history(period="1d", start=start_date, end=end_date)

df = calculate_technical_indicators(df)

# Prepare labels for multiple horizons
horizons = [2, 5, 30, 60, 100, 1000]
for horizon in horizons:
    df[f'Label_{horizon}'] = np.where(df['Close'].shift(-horizon) > df['Close'], 1, 0)

df.dropna(inplace=True)

# Train a model for each horizon and predict
scaler = MinMaxScaler()
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line']
models = {}

for horizon in horizons:
    print(f"Training model for {horizon}-day horizon.")
    scaled_features = scaler.fit_transform(df[features])
    X, y = create_sequences(pd.DataFrame(scaled_features, columns=features), df[f'Label_{horizon}'], time_steps=10)
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)
    evaluation = model.evaluate(X_test, y_test, verbose=0)
    models[horizon] = model
    print(f"Model for {horizon}-day horizon: Loss = {evaluation[0]}, Accuracy = {evaluation[1]}")

# Prepare the most recent sequence for prediction
most_recent_sequence = scaled_features[-10:].reshape(1, 10, len(features))

# Predict stock movement for each horizon using the saved models
for horizon in horizons:
    model = models[horizon]
    prediction = model.predict(most_recent_sequence)
    predicted_class = np.argmax(prediction, axis=1)
    movement = 'Up' if predicted_class[0] == 1 else 'Down'
    print(f"Model predicts the stock price movement for {horizon}-day horizon as: {movement}")
