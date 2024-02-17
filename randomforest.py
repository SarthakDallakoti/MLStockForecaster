import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_RSI(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def calculate_MACD(data, span1=12, span2=26, signal_span=9):
    exp1 = data['Close'].ewm(span=span1, adjust=False).mean()
    exp2 = data['Close'].ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal_span, adjust=False).mean()
    
    data['MACD'] = macd
    data['Signal_Line'] = signal_line
    return data

def fetch_prepare_data(ticker_symbol, start_date='1990-01-01'):
    ticker_data = yf.Ticker(ticker_symbol)
    end_date = datetime.now().strftime('%Y-%m-%d')
    ticker_df = ticker_data.history(period='1d', start=start_date, end=end_date)
    
    ticker_df = ticker_df.drop(columns=['Dividends', 'Stock Splits'])
    ticker_df['Price_Diff'] = ticker_df['Close'].diff()
    ticker_df['Label'] = np.where(ticker_df['Price_Diff'] > 0, 1, 0)

    # Calculate RSI and add it to DataFrame
    ticker_df['RSI'] = calculate_RSI(ticker_df)

    # Calculate MACD and add it to DataFrame
    ticker_df = calculate_MACD(ticker_df)
    print(ticker_df)
    
    ticker_df = ticker_df.dropna()  # Drop rows with NaN values resulting from calculations

    return ticker_df

def train_model(ticker_df):
    X = ticker_df[['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'Signal_Line']]  # Updated to include new features
    y = ticker_df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    y_pred = rf_classifier.predict(X_test)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    
    return rf_classifier

def predict_next_day_movement(rf_classifier, ticker_df):
    last_row_data = ticker_df.iloc[-1][['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'Signal_Line']]
    prediction_features = pd.DataFrame(last_row_data).transpose()
    
    prediction = rf_classifier.predict(prediction_features)
    prediction_result = "up" if prediction[0] == 1 else "down"
    
    return prediction_result

# Main execution
ticker_symbol = '^GSPC'
ticker_df = fetch_prepare_data(ticker_symbol)
print(ticker_df)
rf_classifier = train_model(ticker_df)
prediction_result = predict_next_day_movement(rf_classifier, ticker_df)
print(f"The model predicts that the stock price will go {prediction_result}.")
