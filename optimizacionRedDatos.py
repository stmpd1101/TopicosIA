# ============================================================
# OPTIMIZACIÓN DE RED DE DATOS CON ALGORITMOS BIOINSPIRADOS
# ------------------------------------------------------------
# Algoritmos incluidos:
#   1) Algoritmo Genético (GA)
#   2) PSO discreto
#   3) GWO discreto
#   4) ABC
#   5) AIS
#
# Problema:
#   - Seleccionar rutas para varias demandas origen-destino
#   - Minimizar:
#       * latencia total
#       * desbalance de carga
#       * exceso de capacidad
#       * congestión respecto al umbral
#
# Representación:
#   - Fenotipo: rutas reales usadas por las demandas en la red
#   - Genotipo: vector discreto, donde cada gen indica cuál
#               ruta candidata usa cada demanda
#
# Requisitos:
#   - Python 3.x
#   - No usa librerías externas
# ============================================================

import random
import math
import copy
from collections import defaultdict

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

SEED = 42
random.seed(SEED)

# Pesos de la función objetivo
ALPHA = 1.0     # peso latencia total
BETA = 50.0     # peso desbalance
GAMMA = 1000.0  # peso exceso de capacidad
DELTA = 800.0   # peso congestión por umbral

# Número máximo de rutas candidatas por demanda
K_RUTAS = 5

# Parámetros generales de algoritmos
POBLACION = 30
ITERACIONES = 80

# ============================================================
# DATOS DEL PROBLEMA
# ============================================================
# Red no dirigida.
# Formato:
#   ("N1", "N2"): {"capacidad": 100, "latencia": 10, "umbral": 0.75}
#
# Puedes cambiar estos datos por los de tu tarea.
# ============================================================

ENLACES = {
    ("N1", "N2"): {"capacidad": 100, "latencia": 4, "umbral": 0.75},
    ("N1", "N3"): {"capacidad": 80,  "latencia": 3, "umbral": 0.80},
    ("N2", "N3"): {"capacidad": 60,  "latencia": 2, "umbral": 0.70},
    ("N2", "N4"): {"capacidad": 100, "latencia": 5, "umbral": 0.80},
    ("N3", "N4"): {"capacidad": 90,  "latencia": 4, "umbral": 0.75},
    ("N3", "N5"): {"capacidad": 70,  "latencia": 6, "umbral": 0.70},
    ("N4", "N5"): {"capacidad": 110, "latencia": 3, "umbral": 0.85},
    ("N4", "N6"): {"capacidad": 95,  "latencia": 4, "umbral": 0.80},
    ("N5", "N6"): {"capacidad": 100, "latencia": 2, "umbral": 0.80},
}

# Demandas: (origen, destino, trafico)
DEMANDAS = [
    ("N1", "N6", 30),
    ("N1", "N5", 25),
    ("N2", "N6", 20),
    ("N2", "N5", 18),
    ("N3", "N6", 15),
]

# ============================================================
# UTILIDADES DE RED
# ============================================================

def normalizar_arista(a, b):
    """Convierte una arista a una representación ordenada."""
    return tuple(sorted((a, b)))

def construir_grafo(enlaces):
    """Construye lista de adyacencia."""
    grafo = defaultdict(list)
    for (u, v), datos in enlaces.items():
        grafo[u].append(v)
        grafo[v].append(u)
    return grafo

GRAFO = construir_grafo(ENLACES)

def obtener_datos_arista(u, v):
    """Devuelve el diccionario de datos de una arista."""
    arista = normalizar_arista(u, v)
    return ENLACES[arista]

def costo_latencia_ruta(ruta):
    """Suma la latencia total de una ruta."""
    total = 0
    for i in range(len(ruta) - 1):
        u, v = ruta[i], ruta[i + 1]
        total += obtener_datos_arista(u, v)["latencia"]
    return total

def ruta_a_aristas(ruta):
    """Convierte una ruta [N1, N2, N4] a [(N1,N2),(N2,N4)]."""
    aristas = []
    for i in range(len(ruta) - 1):
        aristas.append(normalizar_arista(ruta[i], ruta[i + 1]))
    return aristas

# ============================================================
# GENERACIÓN DE RUTAS CANDIDATAS
# ============================================================

def enumerar_rutas_simples(grafo, origen, destino, max_profundidad=8):
    """
    Genera rutas simples entre origen y destino por DFS.
    Para redes pequeñas/medianas de tarea funciona bien.
    """
    rutas = []

    def dfs(actual, meta, visitados, camino):
        if len(camino) > max_profundidad:
            return
        if actual == meta:
            rutas.append(camino[:])
            return
        for vecino in grafo[actual]:
            if vecino not in visitados:
                visitados.add(vecino)
                camino.append(vecino)
                dfs(vecino, meta, visitados, camino)
                camino.pop()
                visitados.remove(vecino)

    dfs(origen, destino, {origen}, [origen])
    return rutas

def k_mejores_rutas(origen, destino, k=K_RUTAS):
    """
    Obtiene hasta k rutas candidatas ordenadas por latencia.
    """
    rutas = enumerar_rutas_simples(GRAFO, origen, destino, max_profundidad=8)
    rutas_ordenadas = sorted(rutas, key=costo_latencia_ruta)
    return rutas_ordenadas[:k]

def generar_rutas_candidatas(demandas):
    """
    Para cada demanda, genera sus rutas candidatas.
    """
    candidatas = []
    for (origen, destino, trafico) in demandas:
        rutas = k_mejores_rutas(origen, destino, K_RUTAS)
        if not rutas:
            raise ValueError(f"No hay ruta entre {origen} y {destino}")
        candidatas.append(rutas)
    return candidatas

RUTAS_CANDIDATAS = generar_rutas_candidatas(DEMANDAS)

# ============================================================
# REPRESENTACIÓN DE SOLUCIONES
# ============================================================

def crear_solucion_aleatoria():
    """
    Crea un genotipo aleatorio.
    Cada gen = índice de ruta candidata seleccionada para una demanda.
    """
    solucion = []
    for rutas in RUTAS_CANDIDATAS:
        solucion.append(random.randint(0, len(rutas) - 1))
    return solucion

def decodificar_solucion(solucion):
    """
    Convierte el genotipo en rutas reales (fenotipo).
    """
    rutas_elegidas = []
    for i, idx_ruta in enumerate(solucion):
        ruta = RUTAS_CANDIDATAS[i][idx_ruta]
        rutas_elegidas.append(ruta)
    return rutas_elegidas

# ============================================================
# FUNCIÓN OBJETIVO
# ============================================================

def evaluar_solucion(solucion, verbose=False):
    """
    Evalúa una solución.

    Retorna:
        costo_total, detalles
    """
    rutas = decodificar_solucion(solucion)

    flujo_por_arista = defaultdict(float)
    latencia_total = 0.0

    # 1) Asignar flujo de cada demanda a la ruta seleccionada
    for i, ruta in enumerate(rutas):
        origen, destino, trafico = DEMANDAS[i]
        aristas = ruta_a_aristas(ruta)

        # Latencia multiplicada por tráfico
        latencia_ruta = costo_latencia_ruta(ruta)
        latencia_total += latencia_ruta * trafico

        for arista in aristas:
            flujo_por_arista[arista] += trafico

    # 2) Calcular utilización de enlaces
    utilizaciones = []
    penalizacion_capacidad = 0.0
    penalizacion_congestion = 0.0

    for arista, datos in ENLACES.items():
        capacidad = datos["capacidad"]
        umbral = datos["umbral"]
        flujo = flujo_por_arista[arista]
        uso = flujo / capacidad if capacidad > 0 else float("inf")
        utilizaciones.append(uso)

        exceso_capacidad = max(0.0, flujo - capacidad)
        penalizacion_capacidad += exceso_capacidad ** 2

        exceso_umbral = max(0.0, uso - umbral)
        penalizacion_congestion += exceso_umbral ** 2

    # 3) Desbalance = varianza de utilizaciones
    promedio = sum(utilizaciones) / len(utilizaciones)
    varianza = sum((u - promedio) ** 2 for u in utilizaciones) / len(utilizaciones)

    # 4) Función objetivo
    costo_total = (
        ALPHA * latencia_total
        + BETA * varianza
        + GAMMA * penalizacion_capacidad
        + DELTA * penalizacion_congestion
    )

    detalles = {
        "latencia_total": latencia_total,
        "varianza_uso": varianza,
        "penalizacion_capacidad": penalizacion_capacidad,
        "penalizacion_congestion": penalizacion_congestion,
        "flujo_por_arista": dict(flujo_por_arista),
        "utilizaciones": utilizaciones,
        "rutas": rutas,
    }

    if verbose:
        imprimir_detalles(solucion, costo_total, detalles)

    return costo_total, detalles

def imprimir_detalles(solucion, costo, detalles):
    print("\n====================================================")
    print("MEJOR SOLUCIÓN ENCONTRADA")
    print("====================================================")
    print("Genotipo:", solucion)
    print("\nRutas elegidas por demanda:")
    for i, ruta in enumerate(detalles["rutas"]):
        origen, destino, trafico = DEMANDAS[i]
        print(f"  Demanda {i+1}: {origen} -> {destino}  tráfico={trafico}")
        print(f"    Ruta: {' -> '.join(ruta)}")
        print(f"    Latencia ruta: {costo_latencia_ruta(ruta)}")

    print("\nMétricas:")
    print(f"  Costo total:               {costo:.4f}")
    print(f"  Latencia total:            {detalles['latencia_total']:.4f}")
    print(f"  Varianza de uso:           {detalles['varianza_uso']:.6f}")
    print(f"  Penalización capacidad:    {detalles['penalizacion_capacidad']:.4f}")
    print(f"  Penalización congestión:   {detalles['penalizacion_congestion']:.4f}")

    print("\nFlujo por enlace:")
    for arista in sorted(ENLACES.keys()):
        datos = ENLACES[arista]
        flujo = detalles["flujo_por_arista"].get(arista, 0.0)
        uso = flujo / datos["capacidad"]
        print(
            f"  {arista}: flujo={flujo:.2f}, capacidad={datos['capacidad']}, "
            f"uso={uso:.3f}, umbral={datos['umbral']}"
        )
    print("====================================================\n")

# ============================================================
# OPERADORES AUXILIARES
# ============================================================

def reparar_solucion(solucion):
    """
    Garantiza que cada gen esté dentro del rango válido.
    """
    reparada = []
    for i, gen in enumerate(solucion):
        max_idx = len(RUTAS_CANDIDATAS[i]) - 1
        gen = int(round(gen))
        if gen < 0:
            gen = 0
        if gen > max_idx:
            gen = max_idx
        reparada.append(gen)
    return reparada

def mutar_solucion(solucion, prob_mutacion=0.1):
    """
    Cambia aleatoriamente algunos genes.
    """
    nueva = solucion[:]
    for i in range(len(nueva)):
        if random.random() < prob_mutacion:
            nueva[i] = random.randint(0, len(RUTAS_CANDIDATAS[i]) - 1)
    return nueva

def vecino_aleatorio(solucion):
    """
    Cambia una sola demanda por otra ruta candidata.
    """
    nueva = solucion[:]
    i = random.randint(0, len(nueva) - 1)
    nueva[i] = random.randint(0, len(RUTAS_CANDIDATAS[i]) - 1)
    return nueva

# ============================================================
# 1) ALGORITMO GENÉTICO
# ============================================================

def seleccion_torneo(poblacion, fitnesses, tam_torneo=3):
    mejor = None
    mejor_fit = float("inf")
    for _ in range(tam_torneo):
        idx = random.randint(0, len(poblacion) - 1)
        if fitnesses[idx] < mejor_fit:
            mejor_fit = fitnesses[idx]
            mejor = poblacion[idx][:]
    return mejor

def cruza_un_punto(p1, p2):
    if len(p1) == 1:
        return p1[:], p2[:]
    punto = random.randint(1, len(p1) - 1)
    h1 = p1[:punto] + p2[punto:]
    h2 = p2[:punto] + p1[punto:]
    return h1, h2

def algoritmo_genetico(tam_poblacion=POBLACION, generaciones=ITERACIONES,
                       prob_cruza=0.9, prob_mutacion=0.12, elitismo=2):
    poblacion = [crear_solucion_aleatoria() for _ in range(tam_poblacion)]

    mejor_sol = None
    mejor_fit = float("inf")

    for gen in range(generaciones):
        fitnesses = []
        for ind in poblacion:
            fit, _ = evaluar_solucion(ind)
            fitnesses.append(fit)
            if fit < mejor_fit:
                mejor_fit = fit
                mejor_sol = ind[:]

        # Elitismo
        pares = sorted(zip(poblacion, fitnesses), key=lambda x: x[1])
        nueva_poblacion = [copy.deepcopy(ind) for ind, _ in pares[:elitismo]]

        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, fitnesses)
            padre2 = seleccion_torneo(poblacion, fitnesses)

            if random.random() < prob_cruza:
                h1, h2 = cruza_un_punto(padre1, padre2)
            else:
                h1, h2 = padre1[:], padre2[:]

            h1 = mutar_solucion(h1, prob_mutacion)
            h2 = mutar_solucion(h2, prob_mutacion)
            h1 = reparar_solucion(h1)
            h2 = reparar_solucion(h2)

            nueva_poblacion.append(h1)
            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(h2)

        poblacion = nueva_poblacion

    return mejor_sol, mejor_fit

# ============================================================
# 2) PSO DISCRETO
# ============================================================

def pso_discreto(num_particulas=POBLACION, iteraciones=ITERACIONES,
                 w=0.7, c1=1.6, c2=1.6):
    # Posiciones reales para permitir movimiento tipo PSO
    posiciones = []
    velocidades = []
    pbest = []
    pbest_fit = []

    for _ in range(num_particulas):
        pos = []
        vel = []
        for rutas in RUTAS_CANDIDATAS:
            max_idx = len(rutas) - 1
            pos.append(random.uniform(0, max_idx))
            vel.append(random.uniform(-1, 1))
        posiciones.append(pos)
        velocidades.append(vel)

        sol = reparar_solucion(pos)
        fit, _ = evaluar_solucion(sol)
        pbest.append(pos[:])
        pbest_fit.append(fit)

    gbest = pbest[pbest_fit.index(min(pbest_fit))][:]
    gbest_fit = min(pbest_fit)

    for _ in range(iteraciones):
        for i in range(num_particulas):
            for d in range(len(RUTAS_CANDIDATAS)):
                r1 = random.random()
                r2 = random.random()
                velocidades[i][d] = (
                    w * velocidades[i][d]
                    + c1 * r1 * (pbest[i][d] - posiciones[i][d])
                    + c2 * r2 * (gbest[d] - posiciones[i][d])
                )
                posiciones[i][d] += velocidades[i][d]

            sol = reparar_solucion(posiciones[i])
            fit, _ = evaluar_solucion(sol)

            if fit < pbest_fit[i]:
                pbest[i] = posiciones[i][:]
                pbest_fit[i] = fit

                if fit < gbest_fit:
                    gbest = posiciones[i][:]
                    gbest_fit = fit

    mejor_sol = reparar_solucion(gbest)
    return mejor_sol, gbest_fit

# ============================================================
# 3) GWO DISCRETO
# ============================================================

def gwo_discreto(num_lobos=POBLACION, iteraciones=ITERACIONES):
    lobos = [crear_solucion_aleatoria() for _ in range(num_lobos)]

    for _ in range(iteraciones):
        evaluados = sorted(
            [(lobo, evaluar_solucion(lobo)[0]) for lobo in lobos],
            key=lambda x: x[1]
        )

        alfa = evaluados[0][0][:]
        beta = evaluados[1][0][:] if len(evaluados) > 1 else alfa[:]
        delta = evaluados[2][0][:] if len(evaluados) > 2 else beta[:]

        a = 2 - 2 * (_ / max(1, iteraciones - 1))

        nuevos_lobos = []
        for lobo in lobos:
            nuevo = []
            for d in range(len(lobo)):
                x = lobo[d]

                A1 = 2 * a * random.random() - a
                C1 = 2 * random.random()
                D_alfa = abs(C1 * alfa[d] - x)
                X1 = alfa[d] - A1 * D_alfa

                A2 = 2 * a * random.random() - a
                C2 = 2 * random.random()
                D_beta = abs(C2 * beta[d] - x)
                X2 = beta[d] - A2 * D_beta

                A3 = 2 * a * random.random() - a
                C3 = 2 * random.random()
                D_delta = abs(C3 * delta[d] - x)
                X3 = delta[d] - A3 * D_delta

                valor = (X1 + X2 + X3) / 3
                max_idx = len(RUTAS_CANDIDATAS[d]) - 1
                valor = int(round(valor))
                valor = max(0, min(max_idx, valor))
                nuevo.append(valor)

            # Ligera mutación para evitar estancamiento
            if random.random() < 0.15:
                nuevo = vecino_aleatorio(nuevo)

            nuevos_lobos.append(reparar_solucion(nuevo))

        lobos = nuevos_lobos

    mejor = min(lobos, key=lambda s: evaluar_solucion(s)[0])
    mejor_fit, _ = evaluar_solucion(mejor)
    return mejor, mejor_fit

# ============================================================
# 4) ABC (ARTIFICIAL BEE COLONY)
# ============================================================

def abc(num_fuentes=POBLACION, iteraciones=ITERACIONES, limite=15):
    fuentes = [crear_solucion_aleatoria() for _ in range(num_fuentes)]
    fitness = [evaluar_solucion(f)[0] for f in fuentes]
    intentos = [0] * num_fuentes

    for _ in range(iteraciones):
        # Abejas empleadas
        for i in range(num_fuentes):
            candidata = vecino_aleatorio(fuentes[i])
            fit_cand, _ = evaluar_solucion(candidata)

            if fit_cand < fitness[i]:
                fuentes[i] = candidata
                fitness[i] = fit_cand
                intentos[i] = 0
            else:
                intentos[i] += 1

        # Probabilidades para abejas observadoras
        # Mejor costo => mayor probabilidad
        aptitudes = [1 / (1 + f) for f in fitness]
        suma_apt = sum(aptitudes)
        probabilidades = [a / suma_apt for a in aptitudes]

        # Abejas observadoras
        for _ in range(num_fuentes):
            r = random.random()
            acumulado = 0.0
            idx = 0
            for i, p in enumerate(probabilidades):
                acumulado += p
                if r <= acumulado:
                    idx = i
                    break

            candidata = vecino_aleatorio(fuentes[idx])
            fit_cand, _ = evaluar_solucion(candidata)

            if fit_cand < fitness[idx]:
                fuentes[idx] = candidata
                fitness[idx] = fit_cand
                intentos[idx] = 0
            else:
                intentos[idx] += 1

        # Abejas exploradoras
        for i in range(num_fuentes):
            if intentos[i] >= limite:
                fuentes[i] = crear_solucion_aleatoria()
                fitness[i] = evaluar_solucion(fuentes[i])[0]
                intentos[i] = 0

    mejor_idx = fitness.index(min(fitness))
    return fuentes[mejor_idx], fitness[mejor_idx]

# ============================================================
# 5) AIS (ARTIFICIAL IMMUNE SYSTEM)
# ============================================================

def ais(tam_poblacion=POBLACION, iteraciones=ITERACIONES,
        num_seleccionados=10, factor_clon=4, tasa_mutacion_base=0.3):
    poblacion = [crear_solucion_aleatoria() for _ in range(tam_poblacion)]

    mejor_sol = None
    mejor_fit = float("inf")

    for _ in range(iteraciones):
        evaluados = sorted(
            [(ind, evaluar_solucion(ind)[0]) for ind in poblacion],
            key=lambda x: x[1]
        )

        if evaluados[0][1] < mejor_fit:
            mejor_sol = evaluados[0][0][:]
            mejor_fit = evaluados[0][1]

        seleccionados = [ind for ind, _ in evaluados[:num_seleccionados]]
        clones = []

        # Clonación + hipermutación
        for rank, ind in enumerate(seleccionados):
            # Mejor rank => menos mutación
            for _ in range(factor_clon):
                clon = ind[:]
                tasa_mutacion = tasa_mutacion_base * ((rank + 1) / num_seleccionados)
                clon = mutar_solucion(clon, tasa_mutacion)
                clon = reparar_solucion(clon)
                clones.append(clon)

        # Mezclar población actual y clones
        candidatos = poblacion + clones

        # Selección de los mejores
        candidatos = sorted(
            candidatos,
            key=lambda s: evaluar_solucion(s)[0]
        )

        nueva_poblacion = candidatos[:tam_poblacion // 2]

        # Introducir diversidad
        while len(nueva_poblacion) < tam_poblacion:
            nueva_poblacion.append(crear_solucion_aleatoria())

        poblacion = nueva_poblacion

    return mejor_sol, mejor_fit

# ============================================================
# IMPRESIÓN DE RUTAS CANDIDATAS
# ============================================================

def mostrar_rutas_candidatas():
    print("\n====================================================")
    print("RUTAS CANDIDATAS POR DEMANDA")
    print("====================================================")
    for i, demanda in enumerate(DEMANDAS):
        origen, destino, trafico = demanda
        print(f"\nDemanda {i+1}: {origen} -> {destino}  tráfico={trafico}")
        for j, ruta in enumerate(RUTAS_CANDIDATAS[i]):
            print(f"  Ruta {j}: {' -> '.join(ruta)} | Latencia={costo_latencia_ruta(ruta)}")
    print("====================================================\n")

# ============================================================
# EJECUCIÓN Y COMPARACIÓN
# ============================================================

def ejecutar_todos():
    mostrar_rutas_candidatas()

    resultados = []

    print("Ejecutando Algoritmo Genético...")
    sol_ga, fit_ga = algoritmo_genetico()
    resultados.append(("Algoritmo Genético", sol_ga, fit_ga))

    print("Ejecutando PSO...")
    sol_pso, fit_pso = pso_discreto()
    resultados.append(("PSO", sol_pso, fit_pso))

    print("Ejecutando GWO...")
    sol_gwo, fit_gwo = gwo_discreto()
    resultados.append(("GWO", sol_gwo, fit_gwo))

    print("Ejecutando ABC...")
    sol_abc, fit_abc = abc()
    resultados.append(("ABC", sol_abc, fit_abc))

    print("Ejecutando AIS...")
    sol_ais, fit_ais = ais()
    resultados.append(("AIS", sol_ais, fit_ais))

    resultados.sort(key=lambda x: x[2])

    print("\n====================================================")
    print("RESUMEN FINAL")
    print("====================================================")
    for nombre, sol, fit in resultados:
        print(f"{nombre:20s} -> costo = {fit:.4f}, solución = {sol}")
    print("====================================================")

    mejor_nombre, mejor_sol, mejor_fit = resultados[0]
    print(f"\nMejor algoritmo en esta ejecución: {mejor_nombre}")
    _, detalles = evaluar_solucion(mejor_sol)
    imprimir_detalles(mejor_sol, mejor_fit, detalles)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ejecutar_todos()