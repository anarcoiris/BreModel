import requests
import json
import time
from pathlib import Path

def descargar_datos(market, symbol, intervals, outdir=None):
    if outdir is None:
        outdir = symbol  # Usar el símbolo como nombre de carpeta por defecto
    Path(outdir).mkdir(exist_ok=True)

    end_at = int(time.time())
    start_at = end_at - 90 * 24 * 3600  # Últimos 90 días

    for label, interval in intervals.items():
        if market == "binance":
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': 100
            }
            r = requests.get(url, params=params)
            data = r.json()
            parsed = [
                {
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                } for item in data
            ]
        elif market == "kucoin":
            url = "https://api.kucoin.com/api/v1/market/candles"
            params = {
                'symbol': symbol,
                'type': interval,
                'startAt': start_at,
                'endAt': end_at
            }
            r = requests.get(url, params=params)
            data = r.json().get('data', [])
            parsed = [
                {
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'close': float(item[2]),
                    'high': float(item[3]),
                    'low': float(item[4]),
                    'volume': float(item[5])
                } for item in reversed(data)
            ]
        else:
            continue

        symbol_clean = symbol.replace("-", "")  # Para unificar BTCUSDT y BTC-USDT
        fname = f"{outdir}/{symbol_clean}_{label}_{market}.json"
        with open(fname, 'w') as f:
            json.dump(parsed, f, indent=2)
        print(f"Guardado: {fname}")

# Parámetros
descargar_datos("binance", "XMRUSDT", {"4h": "4h", "1D": "1d", "1W": "1w"})
descargar_datos("kucoin", "XMR-USDT", {"4h": "4hour", "1D": "1day", "1W": "1week"})
descargar_datos("binance", "BTCUSDT", {"4h": "4h", "1D": "1d", "1W": "1w"})
descargar_datos("kucoin", "BTC-USDT", {"4h": "4hour", "1D": "1day", "1W": "1week"})
descargar_datos("binance", "DOGEUSDT", {"4h": "4h", "1D": "1d", "1W": "1w"})
descargar_datos("kucoin", "DOGE-USDT", {"4h": "4hour", "1D": "1day", "1W": "1week"})
