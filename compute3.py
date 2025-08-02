import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def calcular_y_exportar_retorno_normalizado(directorio_base, clave_precio='close', csv_output='retornos.csv'):
    archivos = list(Path(directorio_base).rglob("*.json"))
    all_rows = []

    for archivo in archivos:
        with open(archivo) as f:
            datos = json.load(f)

        if not datos or len(datos) < 2:
            continue

        ts = np.array([item['timestamp'] for item in datos])
        xs = np.array([item[clave_precio] for item in datos])

        dX = np.diff(xs)
        dt = np.diff(ts)

        retornos = dX / dt  # (X_{t+τ} - X_t) / Δτ

        for i in range(len(retornos)):
            all_rows.append({
                'archivo': str(archivo),
                'timestamp': ts[i + 1],  # el tiempo correspondiente a t+τ
                'precio': xs[i + 1],
                'retorno': retornos[i]
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_output, index=False)
    return df

def graficar_pendientes(df_retorno, carpeta_salida="plots"):
    Path(carpeta_salida).mkdir(exist_ok=True)
    archivos_unicos = df_retorno['archivo'].unique()

    for archivo in archivos_unicos:
        ejemplo = df_retorno[df_retorno['archivo'] == archivo].copy()
        x = ejemplo['timestamp'].values
        y = ejemplo['precio'].values

        plt.figure(figsize=(16, 8))
        plt.plot(x, y, 'ro', markersize=4, label='Precios')

        # Dibujar pendientes entre (x_0, y_0) y cada (x_i, y_i)
        for i in range(1, len(x)):
            x0, y0 = x[0], y[0]
            xi, yi = x[i], y[i]
            pendiente = (yi - y0) / (xi - x0) if xi != x0 else 0
            plt.plot([x0, xi], [y0, yi], color='gray', alpha=0.2)

        # Promedio de los próximos 10 puntos (amarillo)
        for i in range(len(x) - 10):
            x_i = x[i]
            y_i = y[i]
            x_10 = x[i+10]
            y_10 = y[i+10]
            pendiente = (y_10 - y_i) / (x_10 - x_i)
            plt.plot([x_i, x_10], [y_i, y_10], color='yellow', lw=2, alpha=0.7)

        # Promedio por bloques de 10 (verde)
        for i in range(0, len(x) - 10, 10):
            x_i = x[i]
            y_i = y[i]
            x_f = x[i + 10]
            y_f = y[i + 10]
            pendiente = (y_f - y_i) / (x_f - x_i)
            plt.plot([x_i, x_f], [y_i, y_f], color='green', lw=3, alpha=0.9)

        plt.xlabel("Timestamp")
        plt.ylabel("Precio")
        plt.title(f"Acción del precio — {Path(archivo).name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)

        # Guardar figura
        nombre_salida = Path(carpeta_salida) / (Path(archivo).stem + "_pendientes.png")
        plt.savefig(nombre_salida)
        plt.close()

# Ejecutar
df_retorno = calcular_y_exportar_retorno_normalizado(".")
graficar_pendientes(df_retorno)