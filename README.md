# 🧠 Gold Price Analysis & Forecasting with Random Forest

This project performs a complete pipeline of financial time series analysis on **Gold Futures (GC=F)** using historical data from Yahoo Finance.

## 🎯 Project Motivation & Objectives

Gold is one of the most actively traded commodities and a key macroeconomic hedge, especially during inflationary and geopolitical uncertainty. The goal of this project is to:

- Understand and quantify the historical performance and volatility of gold
- Build an interpretable and robust machine learning model to forecast gold prices
- Use technical indicators to enhance the predictive power of the model
- Provide actionable visual insights for potential trading strategies

This project serves as a practical application of **financial data science**, **feature engineering**, and **machine learning** for real-world commodity analysis.

## 🔍 Project Highlights

- ✅ Data collection via the `yfinance` API
- 📊 Statistical analysis:
  - Cumulative return
  - Annualized volatility & return
  - Rolling moving average
  - Price/MA20 correlation
- 🌲 Machine Learning:
  - Random Forest Regressor
  - `GridSearchCV` hyperparameter tuning
- 📈 Visualizations of historical performance & model predictions

## 📌 Key Results

| Metric                | Value     |
|----------------------|-----------|
| Cumulative Return     | ~85.86%   |
| Annualized Return     | ~19.54%   |
| Annualized Volatility | ~15.64%   |
| Model R² Score        | ~99.73%   |

## ⚙️ Technical Indicators Used

To build meaningful input features, several widely-used technical indicators were selected:

- **MA20 (20-day Moving Average):**  
  Smooths price data to identify short-term trends and support/resistance levels. The 20-day window is a standard for capturing monthly cycles.

- **STD20 (20-day Standard Deviation):**  
  Measures short-term price volatility. Useful for detecting breakouts or trend exhaustion.

- **RSI (Relative Strength Index):**  
  Captures momentum and potential overbought (>70) or oversold (<30) market conditions over 14 days, as originally designed by Welles Wilder.

- **Momentum:**  
  The difference between current price and the price 10 days ago, to quantify acceleration in price movement.

- **Daily Returns:**  
  Used to compute cumulative performance and risk metrics such as annualized volatility and return.

These indicators collectively capture **trend, volatility, and momentum**—key elements for robust financial modeling.

## 🧰 Technologies Used

- Python 3.x
- [yfinance](https://pypi.org/project/yfinance/)
- pandas
- scikit-learn
- matplotlib

## 📈 Visualizations

### 1. Cumulative Return since 2022
![Cumulative Return](cumulative_return.png)

### 2. Gold Price with 20-Day Moving Average
![Gold Historical](historical.png)

### 3. Gold Price Prediction - Random Forest
![Gold Forecast](prediction.png)

## 🚀 Run it Yourself

1. **Clone the repo**:

```bash
git clone https://github.com/JoLml/gold-forecast-model.git
cd gold-forecast-model
