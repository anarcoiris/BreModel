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
                'timestamp': ts[i+1],  # el tiempo correspondiente a t+τ
                'precio': xs[i+1],
                'retorno': retornos[i]
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_output, index=False)
    return df

# Calcular y exportar
df_retorno = calcular_y_exportar_retorno_normalizado(".")

# Graficar: usar solo una muestra representativa de un archivo para claridad
ejemplo = df_retorno[df_retorno['archivo'] == df_retorno['archivo'].iloc[0]].copy()
x = ejemplo['timestamp'].values
y = ejemplo['precio'].values
u = np.ones_like(x)
v = ejemplo['retorno'].values

# Graficar flechas en (x, y) con dirección (u, v)
plt.figure(figsize=(14, 6))
plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, width=0.002, color='blue')
plt.scatter(x, y, color='red', s=10, label='Precios')
plt.xlabel("Timestamp")
plt.ylabel("Precio")
plt.title("Pendientes normalizadas (retornos) sobre precios")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
