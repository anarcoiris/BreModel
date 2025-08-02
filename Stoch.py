# Reimportar dependencias tras el reset
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# Cargar datos del archivo JSON nuevamente
file_path = "BTCUSDT_1D_binance.json"
with open(file_path, "r") as f:
    raw_data = json.load(f)

# Convertir a DataFrame
df = pd.DataFrame(raw_data)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)
df["close"] = df["close"].astype(float)

# Calcular drift (μ) y volatilidad (σ²)
df["returns"] = df["close"].diff()
mu = df["returns"].mean()
sigma = df["returns"].std()

# Número de simulaciones y horizonte temporal
n_simulations = 100
n_days = len(df) // 2  # Simular hasta T + T/2
last_price = df["close"].iloc[-1]
last_date = df.index[-1]

# Simulaciones de trayectorias futuras con movimiento browniano con drift
dt = 1  # 1 día
paths = np.zeros((n_days, n_simulations))
paths[0] = last_price

for t in range(1, n_days):
    z = np.random.normal(0, 1, n_simulations)
    paths[t] = paths[t - 1] + mu * dt + sigma * np.sqrt(dt) * z

# Crear fechas para el periodo futuro
future_dates = [last_date + timedelta(days=i) for i in range(n_days)]

# Graficar
plt.figure(figsize=(12, 6))
plt.plot(df["close"], label="Histórico", color="black", linewidth=1.2)
for i in range(n_simulations):
    plt.plot(future_dates, paths[:, i], color="red", alpha=0.15)
plt.axvline(x=last_date, linestyle="--", color="gray")
plt.title("Simulación de Trayectorias Futuras (T → T+T/2)")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
