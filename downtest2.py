import time
import requests
import json
from pathlib import Path

def descargar_binance_n_velas(symbol="BTCUSDT", interval="1d", outdir="BTCUSDT_1d", N=52500):
    Path(outdir).mkdir(exist_ok=True)

    url = "https://api.binance.com/api/v3/klines"
    limit_por_llamada = 1000

    now = int(time.time() * 1000)  # timestamp actual en ms

    # Duración de cada vela en milisegundos
    interval_ms = {
        "1m": 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "1d": 24 * 60 * 60_000
    }[interval]

    total_data = []
    end_time = now
    velas_restantes = N

    while velas_restantes > 0:
        cantidad = min(velas_restantes, limit_por_llamada)
        start_time = end_time - cantidad * interval_ms

        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': cantidad
        }

        r = requests.get(url, params=params)
        data = r.json()

        if not data or 'code' in data:
            print("Error al obtener datos:", data)
            break

        # Prependemos porque estamos yendo hacia atrás
        total_data = data + total_data

        end_time = data[0][0]  # retrocedemos al inicio del primer dato recibido
        velas_restantes -= len(data)

        time.sleep(0.1)  # para evitar rate limits

    # Guardar datos procesados
    parsed = [
        {
            'timestamp': int(item[0]),
            'open': float(item[1]),
            'high': float(item[2]),
            'low': float(item[3]),
            'close': float(item[4]),
            'volume': float(item[5])
        } for item in total_data
    ]

    nombre_archivo = f"{symbol}_{interval}_{N}_ultimas_velas.json"
    with open(f"{outdir}/{nombre_archivo}", "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"Guardadas {len(parsed)} velas en {outdir}/{nombre_archivo}")

# Ejemplo de uso
descargar_binance_n_velas("BTCUSDT", interval="1d", outdir="BTCUSDT_1d", N=1500)
