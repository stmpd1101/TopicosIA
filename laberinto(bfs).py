import random
from collections import deque

def crear_tablero(m, n):
    return [["." for _ in range(n)] for _ in range(m)]

def generar_obstaculos(tablero, porcentaje):
    m = len(tablero)
    n = len(tablero[0])
    total_celdas = m * n
    cantidad_obstaculos = int(total_celdas * porcentaje / 100)

    generados = 0
    while generados < cantidad_obstaculos:
        i = random.randint(0, m - 1)
        j = random.randint(0, n - 1)
        if tablero[i][j] == ".":
            tablero[i][j] = "X"
            generados += 1

def imprimir_tablero(tablero):
    for fila in tablero:
        print(" ".join(fila))
    print()

def es_valido(tablero, posicion):
    m = len(tablero)
    n = len(tablero[0])
    x, y = posicion
    return (0 <= x < m) and (0 <= y < n) and (tablero[x][y] != "X")

def bfs(tablero, inicio, meta):
    # Orden de movimientos 
    acciones = [
        ("arriba",    (-1, 0)),
        ("abajo",     ( 1, 0)),
        ("izquierda", ( 0,-1)),
        ("derecha",   ( 0, 1)),
    ]

    # frontera = cola FIFO
    frontera = deque([inicio])
    visitados = set([inicio])

    # Para reconstruir el camino:
    # padre[estado_hijo] = (estado_padre, accion_usada)
    padre = {inicio: (None, None)}

    # while True:
    while frontera:
        nodo = frontera.popleft()  # Pop(frente)

        # if goal_test(nodo): return nodo
        if nodo == meta:
            # reconstruir ruta
            ruta_movs = []
            ruta_pos = []
            cur = meta
            while cur is not None:
                ruta_pos.append(cur)
                prev, accion = padre[cur]
                if accion is not None:
                    ruta_movs.append(accion)
                cur = prev
            ruta_movs.reverse()
            ruta_pos.reverse()
            return ruta_movs, ruta_pos

        # hijos = Generate_solutions(nodo)
        x, y = nodo
        for accion, (dx, dy) in acciones:
            hijo = (x + dx, y + dy)
            if es_valido(tablero, hijo) and hijo not in visitados:
                visitados.add(hijo)
                padre[hijo] = (nodo, accion)
                frontera.append(hijo)  # Insert(hijo, frontera) al final (FIFO)

    # if Empty(frontera): return False
    return None, None

def marcar_ruta(tablero, ruta_pos, inicio, meta):
    # Marca el camino con "*", sin pisar I y M
    for (x, y) in ruta_pos:
        if (x, y) != inicio and (x, y) != meta and tablero[x][y] == ".":
            tablero[x][y] = "*"

# =======================
# PROGRAMA PRINCIPAL
# =======================

m = int(input("Ingrese número de filas (m): "))
n = int(input("Ingrese número de columnas (n): "))
porcentaje = float(input("Ingrese porcentaje de obstáculos: "))

tablero = crear_tablero(m, n)
generar_obstaculos(tablero, porcentaje)

inicio = (0, 0)
meta = (m - 1, n - 1)

# Asegurar que inicio/meta no queden como obstáculo
tablero[inicio[0]][inicio[1]] = "."
tablero[meta[0]][meta[1]] = "."

tablero[inicio[0]][inicio[1]] = "I"
tablero[meta[0]][meta[1]] = "M"

print("\nTablero generado:")
imprimir_tablero(tablero)

ruta_movs, ruta_pos = bfs(tablero, inicio, meta)

if ruta_movs is None:
    print("No existe camino desde I hasta M con esta configuración.")
else:
    print(f"Camino encontrado por BFS con {len(ruta_movs)} pasos.")
    print("Movimientos:", ", ".join(ruta_movs))

    # Mostrar tablero con la ruta
    # (primero quita I y M para marcar limpio, luego los vuelves a poner)
    tablero[inicio[0]][inicio[1]] = "."
    tablero[meta[0]][meta[1]] = "."
    marcar_ruta(tablero, ruta_pos, inicio, meta)
    tablero[inicio[0]][inicio[1]] = "I"
    tablero[meta[0]][meta[1]] = "M"

    print("\nTablero con ruta (*):")
    imprimir_tablero(tablero)

    print("Ruta (posiciones):", ruta_pos)
