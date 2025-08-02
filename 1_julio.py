import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# === Parámetros configurables ===
file_path = "BTCUSDT_1d_1500_ultimas_velas.json"
entrenamiento_dias = 60
test_dias = 30
n_simulaciones = 100

# === 1. Cargar y preparar los datos ===
with open(file_path, "r") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)
df["close"] = df["close"].astype(float)

# Verificación de datos
if len(df) < (entrenamiento_dias + test_dias):
    raise ValueError("No hay suficientes datos para el periodo deseado")

# Selección de ventanas
df_train = df.iloc[-(entrenamiento_dias + test_dias):-test_dias]
df_test = df.iloc[-test_dias:]
last_price = df_train["close"].iloc[-1]
last_date = df_train.index[-1]

# === 2. Cálculo de drift y volatilidad ===
df_train["returns"] = df_train["close"].diff()
mu = df_train["returns"].mean()
sigma = df_train["returns"].std()

# === 3. Simulación de trayectorias ===
n_days = len(df_test)
dt = 1
paths = np.zeros((n_days, n_simulaciones))
paths[0] = last_price

for t in range(1, n_days):
    z = np.random.normal(0, 1, n_simulaciones)
    paths[t] = paths[t - 1] + mu * dt + sigma * np.sqrt(dt) * z

future_dates = [last_date + timedelta(days=i) for i in range(n_days)]

# === 4. Evaluación de precisión ===
mean_pred = paths.mean(axis=1)
mse = np.mean((df_test["close"].values - mean_pred) ** 2)
lower = np.percentile(paths, 10, axis=1)
upper = np.percentile(paths, 90, axis=1)
coverage = np.mean((df_test["close"].values >= lower) & (df_test["close"].values <= upper))

# === 5. Detección de nodos de convergencia ===
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

# === 6. Gráfico ===
plt.figure(figsize=(24, 12))
plt.plot(df_train["close"], label="Entrenamiento", color="black", linewidth=1.2)
colors = plt.cm.rainbow(np.linspace(0, 1, n_simulaciones))
for i in range(n_simulaciones):
    plt.plot(future_dates, paths[:, i], color=colors[i], alpha=0.5, linewidth=1)

plt.plot(df_test["close"], label="Test Real", color="green", linewidth=2)
plt.axvline(x=last_date, linestyle="--", color="gray")

for idx, point in enumerate(top_points):
    plt.scatter(point["date"], point["price"], color="blue", s=80, label=f"Nodo #{idx+1}" if idx == 0 else None)
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
plt.savefig("simulacion_ventana.png")
plt.show()
