import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# === 1. Cargar datos históricos para entrenar simulaciones ===
with open("BTCUSDT_1D_train.json", "r") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)
df["close"] = df["close"].astype(float)

# Calcular drift (μ) y volatilidad (σ²)
df["returns"] = df["close"].diff()
mu = df["returns"].mean()
sigma = df["returns"].std()

# Simular trayectorias futuras desde el último punto
n_simulations = 100
n_days = len(df) // 2
last_price = df["close"].iloc[-1]
last_date = df.index[-1]
dt = 1  # 1 día
paths = np.zeros((n_days, n_simulations))
paths[0] = last_price

for t in range(1, n_days):
    z = np.random.normal(0, 1, n_simulations)
    paths[t] = paths[t - 1] + mu * dt + sigma * np.sqrt(dt) * z

future_dates = [last_date + timedelta(days=i) for i in range(n_days)]

# === 2. Cargar datos reales de test (periodo futuro) ===
with open("BTCUSDT_test_1D.json", "r") as f:
    test_data = json.load(f)

df_test = pd.DataFrame(test_data)
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"], unit="ms")
df_test.set_index("timestamp", inplace=True)
df_test["close"] = df_test["close"].astype(float)

# Alinear las fechas
df_test = df_test[df_test.index >= future_dates[0]]
df_test = df_test.iloc[:n_days]

#n_days = len(df_test) // 2

# === 3. Evaluar precisión de las predicciones ===
mean_pred = paths.mean(axis=1)
mse = np.mean((df_test["close"].values - mean_pred) ** 2)

lower = np.percentile(paths, 10, axis=1)
upper = np.percentile(paths, 90, axis=1)
coverage = np.mean((df_test["close"].values >= lower) & (df_test["close"].values <= upper))

# === 4. Detectar puntos de convergencia ===
n_bins = 100
bin_range = (paths.min(), paths.max())
density_info = []

for t_index, prices_at_t in enumerate(paths):
    hist, bin_edges = np.histogram(prices_at_t, bins=n_bins, range=bin_range)
    max_count = hist.max()
    max_bin_index = hist.argmax()
    bin_center = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])
    time_weight = (t_index + 1) / len(paths)
    weighted_score = max_count * time_weight

    density_info.append({
        "score": weighted_score,
        "count": max_count,
        "time_index": t_index,
        "price": bin_center,
        "date": future_dates[t_index]
    })

top_points = sorted(density_info, key=lambda x: x["score"], reverse=True)[:10]

# === 5. Graficar ===
plt.figure(figsize=(24, 12))
plt.plot(df["close"], label="Histórico", color="black", linewidth=1.2)
colors = plt.cm.rainbow(np.linspace(0, 1, n_simulations))
for i in range(n_simulations):
    plt.plot(future_dates, paths[:, i], color=colors[i], alpha=0.6, linewidth=1)

plt.plot(df_test["close"], label="Real futuro", color="green", linewidth=2)
plt.axvline(x=last_date, linestyle="--", color="gray")

# Marcar nodos
for idx, point in enumerate(top_points):
    plt.scatter(point["date"], point["price"], color="blue", s=80, label=f"Nodo #{idx+1}" if idx==0 else None)
    plt.annotate(
        f"Nodo {idx+1}\n{point['date'].date()}\n{point['price']:.2f} USD",
        (point["date"], point["price"]),
        textcoords="offset points", xytext=(0, 10), ha='center',
        fontsize=8, color='blue'
    )

plt.title(f"Simulación vs Realidad | MSE = {mse:.2f} | Cobertura 10–90% = {coverage*100:.1f}%")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("BTCUSDT_sim_vs_real.png")
plt.show()
