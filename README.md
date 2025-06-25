# Gold Price Analysis & Forecasting with Random Forest

This project performs a full pipeline of financial time series analysis on Gold Futures (`GC=F`), using data from Yahoo Finance. It includes:
- Data collection via the `yfinance` API
- Statistical analysis (cumulative return, volatility, moving average, correlation)
- Random Forest model for price prediction
- GridSearchCV hyperparameter tuning
- Visualizations of historical performance and model predictions

## Key Results
- Cumulative Return: ~85.86%
- Annualized Return: ~19.54%
- Volatility: ~15.64%
- Model R² Score: ~99.73%

## Technologies Used
- Python
- yfinance
- pandas
- scikit-learn
- matplotlib

## Run it yourself
Make sure you have the following libraries:
```
pip install yfinance pandas scikit-learn matplotlib
```

Then run:
```
python main.py
```
