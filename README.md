# MLStockForecaster

#Stock Market Prediction with Machine Learning

##Project Overview
This project explores the application of machine learning techniques to predict stock market movements. Starting with a Random Forest Classifier to predict daily stock price movements, the project evolved to incorporate Long Short-Term Memory (LSTM) networks, acknowledging their potential in capturing the sequential dependencies of time series data like stock prices.

##Transition from Random Forest to LSTM
Initial experiments were conducted with a Random Forest Classifier, leveraging its capability to handle nonlinear data efficiently. While this model provided a foundational understanding, the need to capture temporal patterns in stock prices led to the exploration of LSTM networks, a type of recurrent neural network well-suited for time series analysis.

##Incorporating Prediction Horizons
To enhance the LSTM model's utility for different trading strategies, we integrated prediction horizons ranging from short-term (2, 5 days) to long-term (30, 60, 100, and 1000 days). This approach allowed us to assess the model's performance across various time frames, catering to both short-term traders and long-term investors.

##Findings
###Long-Term Predictions: The LSTM model demonstrated improved accuracy for longer-term horizons (30, 60, 100, and 1000 days), suggesting its effectiveness in capturing and predicting broader market trends.
###Short-Term Challenges: Despite the success in long-term forecasting, predicting short-term market movements remains challenging, highlighting the volatile and unpredictable nature of the stock market in the short run. Efforts to enhance short-term prediction accuracy are ongoing, with explorations into feature engineering, alternative models, and incorporating market sentiment analysis.

##Future Work
###Enhanced Feature Set: Investigate additional features and technical indicators that may improve the model's predictive power, especially for short-term horizons.
###Alternative Models: Experiment with other machine learning and deep learning models that might offer better accuracy for short-term predictions.
###Market Sentiment Analysis: Integrate news and social media sentiment analysis to capture the impact of market sentiment on stock price movements.

##Conclusion
This project underscores the potential of machine learning and deep learning in stock market prediction, with promising results for long-term forecasting. The journey from Random Forest to LSTM and the introduction of prediction horizons illustrate the iterative process of model development and optimization in the quest for better predictive performance.