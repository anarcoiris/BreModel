import json
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

# === SIMULADOR BÁSICO ===
def simular_segmento(segmento, pasos):
    precios = [(vela["open"] + vela["close"]) / 2 for vela in segmento]
    simulados = []
    if not precios:
        return simulados

    mu = (precios[-1] - precios[0]) / len(precios)
    sigma = 0.01 * precios[-1]
    valor_actual = precios[-1]

    for _ in range(pasos):
        drift = mu
        ruido = sigma
        valor_actual += drift
        simulados.append(valor_actual)

    return simulados

# === CARGAR JSON ===
def cargar_datos_json(ruta):
    with open(ruta, 'r') as f:
        return json.load(f)

# === OBTENER SEGMENTO REAL ===
def obtener_segmento_real(datos, indice_inicio, pasos):
    return datos[indice_inicio : indice_inicio + pasos]

# === COMPARAR CON DATOS REALES ===
def comparar_con_datos_reales(precios_simulados, segmento_real):
    precios_reales = [(vela["open"] + vela["close"]) / 2 for vela in segmento_real]

    if not precios_reales or not precios_simulados:
        return {"MSE": None, "Cobertura (>95%)": None}

    min_len = min(len(precios_simulados), len(precios_reales))
    precios_simulados = precios_simulados[:min_len]
    precios_reales = precios_reales[:min_len]

    mse = mean_squared_error(precios_reales, precios_simulados)
    cobertura = sum(1 for p_sim, p_real in zip(precios_simulados, precios_reales)
                    if min(p_sim, p_real) / max(p_sim, p_real) > 0.95) / len(precios_simulados)

    return {
        "MSE": mse,
        "Cobertura (>95%)": cobertura,
    }

# === GRAFICAR ===
def graficar_simulacion(precios_simulados, precios_reales, indice_inicio, par, mercado, temporalidad, ruta_salida):
    plt.figure(figsize=(10, 5))
    plt.plot(precios_simulados, label="Simulado", linestyle='--', color='blue')
    plt.plot(precios_reales, label="Real", linestyle='-', color='green')
    plt.title(f"Simulación vs Real | {par} - {mercado} - {temporalidad} | Desde {indice_inicio}")
    plt.xlabel("Paso")
    plt.ylabel("Precio promedio (Open+Close)/2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(ruta_salida, exist_ok=True)
    plt.savefig(os.path.join(ruta_salida, f"sim_vs_real_{indice_inicio}.png"))
    plt.close()

# === EJECUTAR SIMULACIONES ===
def correr_simulaciones(archivo_json, modo="progresivo", n=10, pasos=50):
    datos = cargar_datos_json(archivo_json)
    nombre_archivo = os.path.splitext(os.path.basename(archivo_json))[0]

    par, temporalidad = nombre_archivo.split("_")[:2]
    mercado = "Desconocido"

    if modo == "progresivo":
        total = len(datos)
        saltos = total // (n + 1)
        indices = [saltos * (i + 1) for i in range(n)]

        for indice in indices:
            if indice + pasos > len(datos):
                print(f"Salteado índice {indice} por falta de datos futuros")
                continue

            datos_hasta_ahora = datos[:indice]
            segmento_real = obtener_segmento_real(datos, indice, pasos)
            precios_simulados = simular_segmento(datos_hasta_ahora, pasos)
            metricas = comparar_con_datos_reales(precios_simulados, segmento_real)

            precios_reales = [(vela["open"] + vela["close"]) / 2 for vela in segmento_real]
            print(f"Simulación desde índice {indice}: {metricas}")

            graficar_simulacion(
                precios_simulados,
                precios_reales,
                indice_inicio=indice,
                par=par,
                mercado=mercado,
                temporalidad=temporalidad,
                ruta_salida=f"resultados/{par}_{temporalidad}"
            )

# === USO PRINCIPAL ===
if __name__ == "__main__":
    archivo_json = "BTCUSDT_1d/BTCUSDT_1d_1500_ultimas_velas.json"

    # Modos disponibles: "divisiones", "progresivo", "combinado"
    correr_simulaciones(archivo_json, modo="progresivo", n=10, pasos=50)
