import json
import numpy as np
from pathlib import Path

def analizar_estadisticas_jsons(directorio_base):
    archivos = list(Path(directorio_base).rglob("*.json"))
    resultados = []

    for archivo in archivos:
        with open(archivo) as f:
            datos = json.load(f)

        if not datos:
            continue

        precios = {key: [] for key in ['open', 'high', 'low', 'close']}

        for entrada in datos:
            for key in precios:
                precios[key].append(entrada.get(key, 0))

        estadisticas = {}
        for key, valores in precios.items():
            arr = np.array(valores)
            media = np.mean(arr)
            sigma = np.std(arr)
            dispersion = sigma / media if media != 0 else 0
            estadisticas[key] = {
                'media': media,
                'sigma': sigma,
                'dispersión': dispersion
            }

        resultados.append({
            'archivo': str(archivo),
            'estadisticas': estadisticas
        })

    return resultados


def calcular_retorno_normalizado(directorio_base, clave_precio='close'):
    archivos = list(Path(directorio_base).rglob("*.json"))
    resultados = []

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

        resultados.append({
            'archivo': str(archivo),
            'retornos': retornos.tolist(),
            'timestamps': ts[1:].tolist()  # coinciden con t+τ
        })

    return resultados


# Estadísticas por archivo
estad = analizar_estadisticas_jsons(".")

for e in estad:
    print(f"\nArchivo: {e['archivo']}")
    for k, v in e['estadisticas'].items():
        print(f"  {k}: media={v['media']:.4f}, σ={v['sigma']:.4f}, dispersión={v['dispersión']:.4f}")

# Retornos normalizados
retornos = calcular_retorno_normalizado(".")
for r in retornos:
    print(f"\nArchivo: {r['archivo']}, primeros retornos: {r['retornos'][:5]}")