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

# Reprocesar simulaciones y marcar los 3 puntos de mayor convergencia ponderada

# Parámetros de binning para contar densidad de trayectorias
n_bins = 100  # número de bins en eje de precio
bin_range = (paths.min(), paths.max())  # rango total de precios simulados

# Guardar las densidades máximas y sus ubicaciones
density_info = []

# Recorremos cada tiempo t y generamos un histograma de precios simulados
for t_index, prices_at_t in enumerate(paths):
    hist, bin_edges = np.histogram(prices_at_t, bins=n_bins, range=bin_range)
    max_count = hist.max()
    max_bin_index = hist.argmax()
    bin_center = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])

    # Ponderamos densidades lejanas al inicio más favorablemente
    time_weight = (t_index + 1) / len(paths)
    weighted_score = max_count * time_weight

    density_info.append({
        "score": weighted_score,
        "count": max_count,
        "time_index": t_index,
        "price": bin_center,
        "date": future_dates[t_index]
    })

# Ordenar por puntuación ponderada y elegir los 3 mejores
top_points = sorted(density_info, key=lambda x: x["score"], reverse=True)[:10]

# Graficar con anotaciones
plt.figure(figsize=(24, 12))
plt.plot(df["close"], label="Histórico", color="black", linewidth=1.2)
colors = plt.cm.rainbow(np.linspace(0, 1, n_simulations))
for i in range(n_simulations):
    plt.plot(future_dates, paths[:, i], color=colors[i], alpha=0.8, linewidth=1)
plt.axvline(x=last_date, linestyle="--", color="gray")

# Marcar los 3 puntos de mayor convergencia
for idx, point in enumerate(top_points):
    plt.scatter(point["date"], point["price"], color="blue", s=80, label=f"Nodo #{idx+1}" if idx==0 else None)
    plt.annotate(
        f"Nodo {idx+1}\n{point['date'].date()}\n{point['price']:.2f} USD",
        (point["date"], point["price"]),
        textcoords="offset points",
        xytext=(0,10),
        ha='center',
        fontsize=8,
        color='blue'
    )

plt.title("Simulación con 10 Nodos de Convergencia de Trayectorias")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("BTCUSDT_1D_binance.png")
plt.show()

# Devolver los 3 puntos para revisión
top_points
