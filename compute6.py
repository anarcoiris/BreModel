import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def animar_pendientes(ts, xs, par, mercado, timeframe, clave_precio, archivo_salida_gif):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(ts, xs, 'ro', markersize=4, label=f'{clave_precio} (puntos)')

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Precio")
    ax.set_title(f"{par.upper()} en {mercado.upper()} ({timeframe}) — {clave_precio}")
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)

    lineas = []

    def init():
        return lineas

    def update(i):
    # Limpiar líneas previas para no acumular en cada frame
        for linea in lineas:
            linea.remove()
        lineas.clear()

    # Dibujar líneas desde el punto i hacia todos los puntos siguientes
        for n in range(i + 1, len(ts)):
            linea, = ax.plot([ts[i], ts[n]], [xs[i], xs[n]], color='yellow', lw=2, alpha=0.7)
            lineas.append(linea)
        return lineas

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(ts) - 10,
        interval=200,  # milisegundos entre cada frame
        blit=True, repeat=False
    )

    ani.save(archivo_salida_gif, writer='pillow', fps=5)
    plt.close(fig)

def obtener_datos_json(archivo, clave_precio):
    with open(archivo) as f:
        datos = json.load(f)
    if not datos or len(datos) < 2:
        return None, None
    ts = np.array([item['timestamp'] for item in datos])
    if clave_precio == 'medio_oc':
        xs = np.array([(item['open'] + item['close']) / 2 for item in datos])
    elif clave_precio == 'medio_hl':
        xs = np.array([(item['high'] + item['low']) / 2 for item in datos])
    else:
        xs = np.array([item[clave_precio] for item in datos])
    return ts, xs

def calcular_retorno(ts, xs):
    dX = np.diff(xs)
    dt = np.diff(ts)
    retornos = dX / dt
    return retornos

def extraer_info_nombre(ruta_archivo):
    nombre = Path(ruta_archivo).stem  # Ej: "DOGEUSDT_4h_binance"
    partes = nombre.split("_")

    if len(partes) == 3:
        par, timeframe, mercado = partes
    else:
        par, timeframe, mercado = "PAR", "timeframe", "MERCADO"

    return par, mercado, timeframe


def graficar_y_guardar(ts, xs, par, mercado, timeframe, clave_precio, archivo_salida):
    plt.figure(figsize=(16, 8))
    plt.plot(ts, xs, 'ro', markersize=4, label=f'{clave_precio} (puntos)')

    # Líneas grises entre el primer punto y los demás
    for i in range(1, len(ts)):
        plt.plot([ts[0], ts[i]], [xs[0], xs[i]], color='gray', alpha=0.2)

    # Promedio de pendientes cada 10 pasos (amarillo)
    for i in range(len(ts) - 10):
        plt.plot([ts[i], ts[i+10]], [xs[i], xs[i+10]], color='yellow', lw=2, alpha=0.7)

    # Pendientes por bloques de 10 (verde)
    for i in range(0, len(ts) - 10, 10):
        plt.plot([ts[i], ts[i+10]], [xs[i], xs[i+10]], color='green', lw=3, alpha=0.9)

    plt.xlabel("Timestamp")
    plt.ylabel("Precio")
    plt.title(f"{par.upper()} en {mercado.upper()} ({timeframe}) — {clave_precio}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(archivo_salida)
    plt.close()

def procesar_directorio(base_dir=".", salida="plots"):
    Path(salida).mkdir(exist_ok=True)
    archivos = list(Path(base_dir).rglob("*.json"))

    for archivo in archivos:
        par, mercado, timeframe = extraer_info_nombre(archivo)

        for clave_precio in ['medio_oc', 'medio_hl']:
            ts, xs = obtener_datos_json(archivo, clave_precio)
            if ts is None:
                continue

            nombre_base = f"{Path(archivo).stem}_{clave_precio}"
            ruta_img = Path(salida) / f"{nombre_base}.png"
            ruta_gif = Path(salida) / f"{nombre_base}.gif"

            # Imagen estática
            graficar_y_guardar(ts, xs, par, mercado, timeframe, clave_precio, ruta_img)

            # Animación GIF
            animar_pendientes(ts, xs, par, mercado, timeframe, clave_precio, ruta_gif)


# Ejecutar
procesar_directorio(".")
