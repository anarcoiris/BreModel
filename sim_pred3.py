import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

def cargar_datos_json(path):
    with open(path, "r") as f:
        return json.load(f)

def simular(subset, nombre_simulacion="default"):
    print(f"Simulando: {nombre_simulacion} con {len(subset)} datos...")

    # Convertir a DataFrame
    df = pd.DataFrame(subset)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df["close"] = df["close"].astype(float)

    # === 1. Calcular drift (μ) y volatilidad (σ) ===
    df["returns"] = df["close"].diff()
    mu = df["returns"].mean()
    sigma = df["returns"].std()

    # === 2. Simular trayectorias ===
    n_simulations = 100
    n_days = len(df) // 2
    last_price = df["close"].iloc[-1]
    last_date = df.index[-1]
    dt = 1
    paths = np.zeros((n_days, n_simulations))
    paths[0] = last_price

    for t in range(1, n_days):
        z = np.random.normal(0, 1, n_simulations)
        paths[t] = paths[t - 1] + mu * dt + sigma * np.sqrt(dt) * z

    future_dates = [last_date + timedelta(days=i) for i in range(n_days)]

    # === 3. Datos reales (test) ===
    if os.path.exists("BTCUSDT_test_1D.json"):
        with open("BTCUSDT_test_1D.json", "r") as f:
            test_data = json.load(f)

        df_test = pd.DataFrame(test_data)
        df_test["timestamp"] = pd.to_datetime(df_test["timestamp"], unit="ms")
        df_test.set_index("timestamp", inplace=True)
        df_test["close"] = df_test["close"].astype(float)

        df_test = df_test[df_test.index >= future_dates[0]]
        df_test = df_test.iloc[:n_days]
        test_close = df_test["close"].values
    else:
        test_close = np.zeros(n_days)

    # === 4. Evaluar precisión ===
    mean_pred = paths.mean(axis=1)
    mse = np.mean((test_close - mean_pred) ** 2) if test_close.any() else 0
    lower = np.percentile(paths, 10, axis=1)
    upper = np.percentile(paths, 90, axis=1)
    coverage = np.mean((test_close >= lower) & (test_close <= upper)) if test_close.any() else 0

    # === 5. Análisis de convergencia por densidad ===
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

    # === 6. Graficar resultados ===
    plt.figure(figsize=(24, 12))
    plt.plot(df["close"], label="Histórico", color="black", linewidth=1.2)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_simulations))
    for i in range(n_simulations):
        plt.plot(future_dates, paths[:, i], color=colors[i], alpha=0.4, linewidth=1)

    if test_close.any():
        plt.plot(df_test["close"], label="Real futuro", color="green", linewidth=2)

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
    return {"nombre": nombre_simulacion, "mse": mse, "cobertura": coverage}


def correr_simulaciones(path_json, modo="progresivo", n=5):
    datos = cargar_datos_json(path_json)
    total = len(datos)

    if total < n:
        raise ValueError("No hay suficientes datos para n divisiones.")

    print(f"Total de datos disponibles: {total}")
    resultados = []

    if modo == "divisiones":
        tamaño = total // n
        for i in range(n):
            inicio = i * tamaño
            fin = (i + 1) * tamaño if i < n - 1 else total
            subset = datos[inicio:fin]
            nombre = f"div_{i+1}_de_{n}"
            resultados.append(simular(subset, nombre))

    elif modo == "progresivo":
        q = total // n
        for i in range(1, n + 1):
            subset = datos[:i * q]
            nombre = f"prog_{i}_q"
            resultados.append(simular(subset, nombre))

    elif modo == "combinado":
        q = total // n
        for i in range(1, n + 1):
            for j in range(n):
                inicio = j * q
                fin = inicio + i * q
                if fin > total:
                    break
                subset = datos[inicio:fin]
                nombre = f"comb_j{j+1}_len{i}q"
                resultados.append(simular(subset, nombre))

    return resultados

# === USO PRINCIPAL ===
if __name__ == "__main__":
    archivo_json = "BTCUSDT_1d/BTCUSDT_1d_1500_ultimas_velas.json"

    # Modos disponibles: "divisiones", "progresivo", "combinado"
    correr_simulaciones(archivo_json, modo="progresivo", n=10)
