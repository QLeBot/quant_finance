"""
Volatilité

La volatilité est une mesure statistique qui quantifie la dispersion des rendements d'un actif ou d'un portefeuille.

Elle est généralement exprimée en pourcentage annuel.

On calcule l'écart-type des rendements logarithmiques journaliers sur une période donnée.
Étapes :
- Récupérer les prix historiques (typiquement les prix de clôture journaliers).
- Calculer les rendements logarithmiques :
    Rendement = ln(Prix_t / Prix_t-1)
- Calculer l'écart-type des rendements :
    Volatilité = écart-type(Rendements) = σ = √(Σ(ln(Prix_t / Prix_t-1) - moyenne(Rendements))^2 / (n-1))
- Annualiser la volatilité :
    Volatilité_annuelle = Volatilité_journalière * √252
"""

# === USING YFINANCE ===
"""
import yfinance as yf
import numpy as np

# Télécharger les données de prix
data = yf.download("AAPL", period="6mo", interval="1d")

# Calcul des rendements log
data['log_ret'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

# Volatilité journalière
vol_journaliere = data['log_ret'].std()

# Volatilité annualisée
vol_annuelle = vol_journaliere * np.sqrt(252)

print(f"Volatilité journalière: {vol_journaliere:.4f}")
print(f"Volatilité annualisée: {vol_annuelle:.4f}")
"""
# === USING ALPACA ===
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

# === Client Alpaca ===
client = StockHistoricalDataClient(API_KEY, API_SECRET)
print(client)

# === Période de récupération ===
end_date = datetime(2023, 12, 31)
#start_date = end_date - timedelta(days=180)
start_date = datetime(2023, 1, 1)   

# === Récupérer les données ===
symbol = 'AAPL'
request_params = StockBarsRequest(
    symbol_or_symbols='AAPL',
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date
)

bars = client.get_stock_bars(request_params).df
print(bars)

# === Nettoyage ===
df = bars[bars.index.get_level_values(0) == symbol].copy()
df.sort_index(inplace=True)

# === Calcul des rendements log ===
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# === Volatilité journalière ===
vol_journaliere = df['log_ret'].std()

# === Volatilité annualisée ===
vol_annuelle = vol_journaliere * np.sqrt(252)

print(f"Volatilité journalière {symbol}: {vol_journaliere:.4%}")
print(f"Volatilité annualisée {symbol}: {vol_annuelle:.4%}")

# === Reset index for plotting ===
df.reset_index(level='symbol', inplace=True)

# === Volatilité ===
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['log_ret'].values, label='Volatilité', color='green')
plt.title(f"Volatilité - {symbol}")
plt.xlabel("Date")
plt.ylabel("Volatilité")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Volatilité glissante ===
df['rolling_vol'] = df['log_ret'].rolling(window=21).std() * np.sqrt(252)  # 1 mois glissant

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['rolling_vol'], label='Volatilité glissante (21 jours)', color='orange')
plt.title(f"Volatilité annualisée glissante - {symbol}")
plt.xlabel("Date")
plt.ylabel("Volatilité annualisée")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

