import time
import requests
import json
from pathlib import Path

def descargar_binance_por_periodos(symbol, interval="1d", outdir="BTCUSDT"):
    Path(outdir).mkdir(exist_ok=True)

    now = int(time.time() * 1000)  # timestamp actual en ms
    un_dia = 24 * 3600 * 1000

    # Entrenamiento: desde hace 120 días hasta hace 60 días
    start_train = now - 120 * un_dia
    end_train = now - 60 * un_dia

    # Test: últimos 30 días
    start_test = end_train
    end_test = now

    def fetch_binance(symbol, interval, start_ms, end_ms, file_name):
        url = "https://api.binance.com/api/v3/klines"
        all_data = []
        while start_ms < end_ms:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 1000
            }
            r = requests.get(url, params=params)
            data = r.json()
            if not data:
                break
            all_data.extend(data)
            start_ms = data[-1][0] + un_dia  # avanzar al día siguiente

        # Guardar a JSON limpio
        parsed = [
            {
                'timestamp': int(item[0]),
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5])
            } for item in all_data
        ]
        with open(f"{outdir}/{file_name}", "w") as f:
            json.dump(parsed, f, indent=2)

    fetch_binance(symbol, interval, start_train, end_train, "BTCUSDT_1D_train.json")
    fetch_binance(symbol, interval, start_test, end_test, "BTCUSDT_1D_test.json")

# Ejemplo de uso:
descargar_binance_por_periodos("BTCUSDT", interval="1d", outdir="BTCUSDT_1d")
