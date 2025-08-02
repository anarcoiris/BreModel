import json
import os

def cargar_datos_json(path):
    with open(path, "r") as f:
        return json.load(f)

def simular(datos, nombre_simulacion="default"):
    # Aquí iría tu simulador
    print(f"Simulación '{nombre_simulacion}' con {len(datos)} datos.")
    # Ejemplo de uso real:
    # resultado = tu_modelo.simular(datos)
    # return resultado

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

    else:
        raise ValueError("Modo inválido: usa 'divisiones', 'progresivo' o 'combinado'.")

    return resultados

# USO:
archivo_json = "BTCUSDT_1d/BTCUSDT_1d_1500_ultimas_velas.json"

# Opciones de modo: "divisiones", "progresivo", "combinado"
correr_simulaciones(archivo_json, modo="progresivo", n=10)
