import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# === Configuraciones principales ===
FILE_PATH = "BTCUSDT_1d_1500_ultimas_velas.json"
ENTRENAMIENTO_DIAS = 60
TEST_DIAS = 30
N_SIMULACIONES = 100

# === 1. Cargar y preparar los datos ===
def cargar_datos_json(path):
    with open(path, "r") as f:
        return json.load(f)

def preparar_dataframe(data):
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df["close"] = df["close"].astype(float)
    return df

# === 2. Simular trayectorias y evaluar ===
def simular(df_total, nombre_simulacion="default"):
    if len(df_total) < ENTRENAMIENTO_DIAS + TEST_DIAS:
        raise ValueError("No hay suficientes datos para entrenamiento + test.")

    df_train = df_total.iloc[-(ENTRENAMIENTO_DIAS + TEST_DIAS):-TEST_DIAS].copy()
    df_test = df_total.iloc[-TEST_DIAS:].copy()

    # === Cálculo de drift y volatilidad ===
    df_train["returns"] = df_train["close"].diff()
    mu = df_train["returns"].mean()
    sigma = df_train["returns"].std()
    last_price = df_train["close"].iloc[-1]
    last_date = df_train.index[-1]

    # === Simulaciones ===
    n_days = len(df_test)
    dt = 1
    paths = np.zeros((n_days, N_SIMULACIONES))
    paths[0] = last_price

    for t in range(1, n_days):
        z = np.random.normal(0, 1, N_SIMULACIONES)
        paths[t] = paths[t - 1] + mu * dt + sigma * np.sqrt(dt) * z

    future_dates = [last_date + timedelta(days=i) for i in range(n_days)]

    # === Evaluar precisión ===
    mean_pred = paths.mean(axis=1)
    test_close = df_test["close"].values
    mse = np.mean((test_close - mean_pred) ** 2)
    lower = np.percentile(paths, 10, axis=1)
    upper = np.percentile(paths, 90, axis=1)
    coverage = np.mean((test_close >= lower) & (test_close <= upper))

    # === Análisis de convergencia ===
    density_info = []
    n_bins = 100
    bin_range = (paths.min(), paths.max())

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

    # === Gráfico ===
    plt.figure(figsize=(24, 12))
    plt.plot(df_train["close"], label="Entrenamiento", color="black", linewidth=1.2)
    colors = plt.cm.rainbow(np.linspace(0, 1, N_SIMULACIONES))
    for i in range(N_SIMULACIONES):
        plt.plot(future_dates, paths[:, i], color=colors[i], alpha=0.4, linewidth=1)

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

    plt.title(f"{nombre_simulacion} | MSE = {mse:.2f} | Cobertura 10–90% = {coverage*100:.1f}%")
    plt.xlabel("Fecha")
    plt.ylabel("Precio (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = f"{nombre_simulacion}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Guardado: {output_path}")
    return {"nombre": nombre_simulacion, "mse": mse, "cobertura": coverage, "mu": mu, "sigma": sigma}

# === USO ===
if __name__ == "__main__":
    datos_json = cargar_datos_json(FILE_PATH)
    df_total = preparar_dataframe(datos_json)
    resultado = simular(df_total, nombre_simulacion="simulacion_ventana_unificada")
    print(resultado)
