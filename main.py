# Visualisation 1 : Historique des prix avec MA20
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Gold_Close'], label="Gold Price", color="black")
plt.plot(data.index, data['MA20'], label="MA20", color="orange", linestyle="--")
plt.title("Gold Price with 20-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("historical.png")
plt.show()

# Visualisation 2 : Prédiction
plt.figure(figsize=(14, 6))
plt.plot(data.index[-len(y_pred):], y[-len(y_pred):], label='Actual Price', color='black')
plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted Price', color='green', linestyle='--')
plt.title("Gold Price Prediction - Random Forest")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("prediction.png")
plt.show()

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# Télécharger les données
ticker = "GC=F"
data = yf.download(ticker, start="2022-01-01", end="2025-06-25")

# Corriger MultiIndex si besoin
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Préparation des données
data = data[["Close"]].dropna()
data.rename(columns={"Close": "Gold_Close"}, inplace=True)

# Ajout d'indicateurs
data["MA20"] = data["Gold_Close"].rolling(20).mean()
data["STD20"] = data["Gold_Close"].rolling(20).std()
data["RSI"] = 100 - (100 / (1 + data["Gold_Close"].pct_change().apply(lambda x: max(x, 0)).rolling(14).mean() /
                             data["Gold_Close"].pct_change().apply(lambda x: -min(x, 0)).rolling(14).mean()))
data["Momentum"] = data["Gold_Close"] - data["Gold_Close"].shift(10)
data["Daily_Return"] = data["Gold_Close"].pct_change()
data.dropna(inplace=True)

# Statistiques descriptives
cumulative_return = (1 + data["Daily_Return"]).cumprod() - 1
annualized_return = data["Daily_Return"].mean() * 252
annualized_vol = data["Daily_Return"].std() * np.sqrt(252)
correlation = data[["Gold_Close", "MA20"]].dropna().corr().iloc[0, 1]

print("📊 Analyse statistique de l'Or (GC=F)")
print(f"Performance cumulée : {cumulative_return[-1]:.2%}")
print(f"Rendement annualisé : {annualized_return:.2%}")
print(f"Volatilité annualisée : {annualized_vol:.2%}")
print(f"Corrélation prix / MA20 : {correlation:.2f}")

# Visualisation 1 : Retour cumulé
plt.figure(figsize=(14, 5))
plt.plot(cumulative_return, label="Cumulative Return", color="gold")
plt.title("Gold - Cumulative Return since 2022")
plt.xlabel("Date")
plt.ylabel("Return (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cumulative_return.png")  # ✅ Enregistrement ajouté
plt.show()

# Target
data["Target"] = data["Gold_Close"].shift(-1)
for i in range(1, 6):
    data[f"Lag_{i}"] = data["Gold_Close"].shift(i)
data.dropna(inplace=True)

# Features & Target
features = [f"Lag_{i}" for i in range(1, 6)] + ["MA20", "STD20", "RSI", "Momentum"]
X = data[features]
y = data["Target"]

# Split avec TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# GridSearch
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

model = RandomForestRegressor(random_state=42)
grid = GridSearchCV(model, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X, y)

print("\n✅ Best parameters from GridSearchCV :")
print(grid.best_params_)

# Prédiction avec meilleur modèle
best_model = grid.best_estimator_
y_pred = best_model.predict(X)

# Évaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"\n🎯 Random Forest Model Evaluation")
print(f"MSE : {mse:.2f}")
print(f"R² : {r2:.2%}")

# Visualisation 2 : Historique des prix avec MA20
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Gold_Close'], label="Gold Price", color="black")
plt.plot(data.index, data['MA20'], label="MA20", color="orange", linestyle="--")
plt.title("Gold Price with 20-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("historical.png")
plt.show()

# Visualisation 3 : Prédiction
plt.figure(figsize=(14, 6))
plt.plot(data.index[-len(y_pred):], y[-len(y_pred):], label='Actual Price', color='black')
plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted Price', color='green', linestyle='--')
plt.title("Gold Price Prediction - Random Forest")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("prediction.png")
plt.show()

