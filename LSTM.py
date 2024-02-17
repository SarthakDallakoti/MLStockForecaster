import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Define functions for RSI and MACD calculations
def calculate_RSI(data, window=14):
    delta = data['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones((window,)) / window, mode='valid')
    avg_loss = np.convolve(loss, np.ones((window,)) / window, mode='valid')
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(window-1, np.nan), rsi])  # Pad with NaN for missing values
    return rsi

def calculate_MACD(data, span1=12, span2=26, signal_span=9):
    exp1 = data['Close'].ewm(span=span1, adjust=False).mean()
    exp2 = data['Close'].ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, signal_line

# Fetch and prepare data
def fetch_prepare_data(ticker_symbol, start_date='1990-01-01'):
    ticker_data = yf.Ticker(ticker_symbol)
    end_date = datetime.now().strftime('%Y-%m-%d')
    ticker_df = ticker_data.history(period='1d', start=start_date, end=end_date)
    ticker_df['RSI'] = calculate_RSI(ticker_df)
    macd, signal_line = calculate_MACD(ticker_df)
    ticker_df['MACD'] = macd
    ticker_df['Signal_Line'] = signal_line
    ticker_df['Label'] = np.where(ticker_df['Close'].diff() > 0, 1, 0)  # 1 if price went up, 0 otherwise
    ticker_df = ticker_df.dropna()  # Drop rows with NaN values
    return ticker_df

# Sequence creation and data preprocessing
def create_sequences_and_preprocess(data, sequence_length=10):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line']])
    
    sequences = []
    labels = []
    for i in range(sequence_length, len(data_scaled)):
        sequences.append(data_scaled[i-sequence_length:i])
        labels.append(data['Label'].iloc[i])
    return np.array(sequences), np.array(labels)

# Main execution
ticker_symbol = '^GSPC'
data = fetch_prepare_data(ticker_symbol)
X, y = create_sequences_and_preprocess(data)
y = to_categorical(y)  # Convert labels to one-hot encoding for binary classification

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM model building
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
evaluation = model.evaluate(X_test, y_test, verbose=0)
print(f'Model Evaluation\n  Loss: {evaluation[0]}\n  Accuracy: {evaluation[1]}')

# Predict the latest sequence
latest_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
prediction = model.predict(latest_sequence)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted Stock Movement: {'Up' if predicted_class[0] == 1 else 'Down'}")
