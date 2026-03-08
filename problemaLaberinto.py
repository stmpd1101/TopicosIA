import random
import math
import heapq
from collections import deque


# =========================================================
# FUNCIONES GENERALES DEL LABERINTO
# =========================================================

def crear_tablero(filas, columnas):
    return [["." for _ in range(columnas)] for _ in range(filas)]


def imprimir_tablero(tablero):
    for fila in tablero:
        print(" ".join(fila))
    print()


def colocar_inicio_meta_aleatorio(filas, columnas):
    inicio = (random.randint(0, filas - 1), random.randint(0, columnas - 1))
    meta = (random.randint(0, filas - 1), random.randint(0, columnas - 1))

    while meta == inicio:
        meta = (random.randint(0, filas - 1), random.randint(0, columnas - 1))

    return inicio, meta


def generar_obstaculos(tablero, porcentaje, inicio, meta):
    filas = len(tablero)
    columnas = len(tablero[0])
    total_celdas = filas * columnas
    cantidad_obstaculos = int(total_celdas * porcentaje / 100)

    generados = 0
    while generados < cantidad_obstaculos:
        x = random.randint(0, filas - 1)
        y = random.randint(0, columnas - 1)

        if tablero[x][y] == "." and (x, y) != inicio and (x, y) != meta:
            tablero[x][y] = "X"
            generados += 1


def es_valido(tablero, posicion):
    filas = len(tablero)
    columnas = len(tablero[0])
    x, y = posicion

    return 0 <= x < filas and 0 <= y < columnas and tablero[x][y] != "X"


def obtener_vecinos(tablero, nodo):
    x, y = nodo
    movimientos = [
        ("arriba", (-1, 0)),
        ("abajo", (1, 0)),
        ("izquierda", (0, -1)),
        ("derecha", (0, 1))
    ]

    vecinos = []
    for nombre, (dx, dy) in movimientos:
        nuevo = (x + dx, y + dy)
        if es_valido(tablero, nuevo):
            vecinos.append((nombre, nuevo))

    return vecinos


def heuristica_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruir_camino(padres, inicio, meta):
    camino = []
    movimientos = []
    actual = meta

    if meta not in padres:
        return None, None

    while actual is not None:
        camino.append(actual)
        padre, accion = padres[actual]
        if accion is not None:
            movimientos.append(accion)
        actual = padre

    camino.reverse()
    movimientos.reverse()
    return camino, movimientos


def marcar_camino(tablero, camino, inicio, meta):
    copia = [fila[:] for fila in tablero]

    for x, y in camino:
        if (x, y) != inicio and (x, y) != meta and copia[x][y] == ".":
            copia[x][y] = "*"

    copia[inicio[0]][inicio[1]] = "I"
    copia[meta[0]][meta[1]] = "M"
    return copia


def mostrar_paso(numero_paso, actual, estructura, visitados):
    print(f"Paso {numero_paso}")
    print(f"Actual: {actual}")
    print(f"Estructura: {estructura}")
    print(f"Visitados: {visitados}")
    print("-" * 50)


# =========================================================
# BFS
# =========================================================

def bfs(tablero, inicio, meta, mostrar=False):
    frontera = deque([inicio])
    visitados = {inicio}
    padres = {inicio: (None, None)}
    paso = 0

    while frontera:
        actual = frontera.popleft()
        paso += 1

        if mostrar:
            mostrar_paso(paso, actual, list(frontera), list(visitados))

        if actual == meta:
            return reconstruir_camino(padres, inicio, meta)

        for accion, vecino in obtener_vecinos(tablero, actual):
            if vecino not in visitados:
                visitados.add(vecino)
                padres[vecino] = (actual, accion)
                frontera.append(vecino)

    return None, None


# =========================================================
# DFS
# =========================================================

def dfs(tablero, inicio, meta, mostrar=False):
    frontera = [inicio]
    visitados = set()
    padres = {inicio: (None, None)}
    paso = 0

    while frontera:
        actual = frontera.pop()
        paso += 1

        if actual in visitados:
            continue

        visitados.add(actual)

        if mostrar:
            mostrar_paso(paso, actual, frontera[:], list(visitados))

        if actual == meta:
            return reconstruir_camino(padres, inicio, meta)

        vecinos = obtener_vecinos(tablero, actual)
        vecinos.reverse()

        for accion, vecino in vecinos:
            if vecino not in visitados and vecino not in frontera:
                padres[vecino] = (actual, accion)
                frontera.append(vecino)

    return None, None


# =========================================================
# LDFS
# =========================================================

def ldfs(tablero, inicio, meta, limite, mostrar=False):
    frontera = [(inicio, 0)]
    visitados = set()
    padres = {inicio: (None, None)}
    paso = 0

    while frontera:
        actual, profundidad = frontera.pop()
        paso += 1

        if actual in visitados:
            continue

        visitados.add(actual)

        if mostrar:
            mostrar_paso(paso, (actual, profundidad), frontera[:], list(visitados))

        if actual == meta:
            return reconstruir_camino(padres, inicio, meta)

        if profundidad < limite:
            vecinos = obtener_vecinos(tablero, actual)
            vecinos.reverse()

            for accion, vecino in vecinos:
                if vecino not in visitados:
                    if vecino not in padres:
                        padres[vecino] = (actual, accion)
                    frontera.append((vecino, profundidad + 1))

    return None, None


# =========================================================
# ILDFS
# =========================================================

def ildfs(tablero, inicio, meta, mostrar=False):
    max_limite = len(tablero) * len(tablero[0])

    for limite in range(max_limite + 1):
        if mostrar:
            print(f"\nIntentando con límite = {limite}\n")
        camino, movimientos = ldfs(tablero, inicio, meta, limite, mostrar)
        if camino is not None:
            return camino, movimientos

    return None, None


# =========================================================
# VORAZ
# =========================================================

def voraz(tablero, inicio, meta, mostrar=False):
    heap = []
    heapq.heappush(heap, (heuristica_manhattan(inicio, meta), inicio))
    visitados = set()
    padres = {inicio: (None, None)}
    paso = 0

    while heap:
        _, actual = heapq.heappop(heap)
        paso += 1

        if actual in visitados:
            continue

        visitados.add(actual)

        if mostrar:
            mostrar_paso(paso, actual, [elem[1] for elem in heap], list(visitados))

        if actual == meta:
            return reconstruir_camino(padres, inicio, meta)

        for accion, vecino in obtener_vecinos(tablero, actual):
            if vecino not in visitados:
                if vecino not in padres:
                    padres[vecino] = (actual, accion)
                h = heuristica_manhattan(vecino, meta)
                heapq.heappush(heap, (h, vecino))

    return None, None


# =========================================================
# A*
# =========================================================

def a_estrella(tablero, inicio, meta, mostrar=False):
    heap = []
    h_inicial = heuristica_manhattan(inicio, meta)
    heapq.heappush(heap, (h_inicial, 0, inicio))

    padres = {inicio: (None, None)}
    costo_g = {inicio: 0}
    visitados = set()
    paso = 0

    while heap:
        _, g_actual, actual = heapq.heappop(heap)
        paso += 1

        if actual in visitados:
            continue

        visitados.add(actual)

        if mostrar:
            mostrar_paso(paso, actual, [elem[2] for elem in heap], list(visitados))

        if actual == meta:
            return reconstruir_camino(padres, inicio, meta)

        for accion, vecino in obtener_vecinos(tablero, actual):
            nuevo_g = g_actual + 1

            if vecino not in costo_g or nuevo_g < costo_g[vecino]:
                costo_g[vecino] = nuevo_g
                padres[vecino] = (actual, accion)
                h = heuristica_manhattan(vecino, meta)
                f = nuevo_g + h
                heapq.heappush(heap, (f, nuevo_g, vecino))

    return None, None


# =========================================================
# BÚSQUEDA TABÚ
# =========================================================

def busqueda_tabu(tablero, inicio, meta, iteraciones_max=100, tamano_tabu=10, mostrar=False):
    actual = inicio
    mejor = inicio
    padres = {inicio: (None, None)}
    lista_tabu = deque(maxlen=tamano_tabu)
    lista_tabu.append(actual)

    for paso in range(1, iteraciones_max + 1):
        if actual == meta:
            return reconstruir_camino(padres, inicio, meta)

        vecinos = obtener_vecinos(tablero, actual)
        if not vecinos:
            return None, None

        candidatos = []
        for accion, vecino in vecinos:
            valor = heuristica_manhattan(vecino, meta)
            candidatos.append((valor, accion, vecino))

        candidatos.sort(key=lambda x: x[0])

        siguiente = None
        accion_elegida = None

        for valor, accion, vecino in candidatos:
            if vecino not in lista_tabu or valor < heuristica_manhattan(mejor, meta):
                siguiente = vecino
                accion_elegida = accion
                break

        if siguiente is None:
            return None, None

        if siguiente not in padres:
            padres[siguiente] = (actual, accion_elegida)

        actual = siguiente
        lista_tabu.append(actual)

        if heuristica_manhattan(actual, meta) < heuristica_manhattan(mejor, meta):
            mejor = actual

        if mostrar:
            mostrar_paso(paso, actual, list(lista_tabu), [])

    if actual == meta:
        return reconstruir_camino(padres, inicio, meta)

    return None, None


# =========================================================
# RECOCIDO SIMULADO
# =========================================================

def recocido_simulado(tablero, inicio, meta, temp_inicial=50, enfriamiento=0.95, iteraciones=200, mostrar=False):
    actual = inicio
    mejor = inicio
    padres = {inicio: (None, None)}
    temperatura = temp_inicial

    for paso in range(1, iteraciones + 1):
        if actual == meta:
            return reconstruir_camino(padres, inicio, meta)

        vecinos = obtener_vecinos(tablero, actual)
        if not vecinos:
            break

        accion, vecino = random.choice(vecinos)

        energia_actual = heuristica_manhattan(actual, meta)
        energia_vecino = heuristica_manhattan(vecino, meta)
        delta = energia_vecino - energia_actual

        aceptar = False

        if delta < 0:
            aceptar = True
        else:
            probabilidad = math.exp(-delta / temperatura) if temperatura > 0 else 0
            if random.random() < probabilidad:
                aceptar = True

        if aceptar:
            if vecino not in padres:
                padres[vecino] = (actual, accion)
            actual = vecino

        if heuristica_manhattan(actual, meta) < heuristica_manhattan(mejor, meta):
            mejor = actual

        if mostrar:
            print(f"Paso {paso} | Actual: {actual} | Mejor: {mejor} | Temperatura: {temperatura:.4f}")

        temperatura *= enfriamiento

        if temperatura < 0.01:
            break

    if actual == meta:
        return reconstruir_camino(padres, inicio, meta)

    return None, None


# =========================================================
# MENÚ Y EJECUCIÓN
# =========================================================

def mostrar_menu():
    print("=== LABERINTO CON ALGORITMOS DE BÚSQUEDA ===")
    print("1. BFS")
    print("2. DFS")
    print("3. LDFS")
    print("4. ILDFS")
    print("5. Voraz")
    print("6. A*")
    print("7. Búsqueda Tabú")
    print("8. Recocido Simulado")
    print("0. Salir")


def ejecutar_algoritmo(opcion, tablero, inicio, meta, mostrar_pasos):
    if opcion == 1:
        return bfs(tablero, inicio, meta, mostrar_pasos), "BFS"

    elif opcion == 2:
        return dfs(tablero, inicio, meta, mostrar_pasos), "DFS"

    elif opcion == 3:
        limite = int(input("Ingrese el límite de profundidad para LDFS: "))
        return ldfs(tablero, inicio, meta, limite, mostrar_pasos), "LDFS"

    elif opcion == 4:
        return ildfs(tablero, inicio, meta, mostrar_pasos), "ILDFS"

    elif opcion == 5:
        return voraz(tablero, inicio, meta, mostrar_pasos), "Voraz"

    elif opcion == 6:
        return a_estrella(tablero, inicio, meta, mostrar_pasos), "A*"

    elif opcion == 7:
        return busqueda_tabu(tablero, inicio, meta, mostrar=mostrar_pasos), "Búsqueda Tabú"

    elif opcion == 8:
        return recocido_simulado(tablero, inicio, meta, mostrar=mostrar_pasos), "Recocido Simulado"

    else:
        return (None, None), "Opción inválida"


def menu_principal():
    print("=== CONFIGURACIÓN DEL LABERINTO ===")
    filas = int(input("Ingrese el número de filas: "))
    columnas = int(input("Ingrese el número de columnas: "))
    porcentaje = float(input("Ingrese el porcentaje de obstáculos: "))

    tablero = crear_tablero(filas, columnas)
    inicio, meta = colocar_inicio_meta_aleatorio(filas, columnas)
    generar_obstaculos(tablero, porcentaje, inicio, meta)

    tablero[inicio[0]][inicio[1]] = "I"
    tablero[meta[0]][meta[1]] = "M"

    print("\nTablero generado:")
    imprimir_tablero(tablero)

    mostrar_menu()
    opcion = int(input("Seleccione un algoritmo: "))

    if opcion == 0:
        print("Programa finalizado.")
        return

    mostrar_pasos = input("¿Desea mostrar el proceso paso a paso? (s/n): ").strip().lower() == "s"

    # Copia del tablero sin alterar la versión original al resolver
    tablero_busqueda = [fila[:] for fila in tablero]

    # Para lógica interna, dejamos I y M transitables
    tablero_busqueda[inicio[0]][inicio[1]] = "."
    tablero_busqueda[meta[0]][meta[1]] = "."

    (camino, movimientos), nombre = ejecutar_algoritmo(opcion, tablero_busqueda, inicio, meta, mostrar_pasos)

    if camino is None:
        print(f"\n{name}: no encontró un camino.")
    else:
        print(f"\n{name}: sí encontró un camino.")
        print("Movimientos:")
        print(movimientos)
        print("\nCamino:")
        print(camino)

        tablero_final = marcar_camino(tablero_busqueda, camino, inicio, meta)

        print("\nTablero con camino encontrado:")
        imprimir_tablero(tablero_final)


if __name__ == "__main__":
    menu_principal()