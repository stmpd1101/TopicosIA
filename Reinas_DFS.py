import time
import math
import random
from collections import deque
import heapq

# ==========================================
# FUNCIONES DE DIBUJO Y VISUALIZACIÓN
# ==========================================
def imprimir_tablero_simple(estado, n):
    for fila in range(n):
        fila_str = ""
        for col in range(n):
            if col < len(estado) and estado[col] == fila:
                fila_str += "[Q] "
            else:
                fila_str += " .  "
        print(fila_str)
    print()

def imprimir_paso(paso, estado, n, tipo="Constructivo"):
    print(f"\n--- Paso {paso} ---")
    if tipo == "Constructivo":
        print(f"Avance exitoso: Reina colocada en columna {len(estado)-1}, fila {estado[-1] if estado else 'N/A'}")
    else:
        conflictos = contar_conflictos(estado)
        print(f"Mejora encontrada - Conflictos actuales: {conflictos}")

    print(f"Estado interno: {estado}")
    imprimir_tablero_simple(estado, n)
    time.sleep(0.02) # Animación fluida

# ==========================================
# LÓGICA DE ESTADOS CONSTRUCTIVOS
# ==========================================
def es_valido(estado):
    if not estado: return True
    ultima_col = len(estado) - 1
    ultima_fila = estado[-1]
    for col in range(ultima_col):
        fila = estado[col]
        if fila == ultima_fila or abs(fila - ultima_fila) == abs(col - ultima_col):
            return False
    return True

# ==========================================
# LÓGICA DE ESTADOS LOCALES (Tabú / Recocido)
# ==========================================
def contar_conflictos(estado):
    conflictos = 0
    n = len(estado)
    for i in range(n):
        for j in range(i + 1, n):
            if estado[i] == estado[j] or abs(estado[i] - estado[j]) == abs(i - j):
                conflictos += 1
    return conflictos

def generar_vecinos_locales(estado):
    vecinos = []
    n = len(estado)
    for col in range(n):
        for fila in range(n):
            if estado[col] != fila:
                nuevo_estado = list(estado)
                nuevo_estado[col] = fila
                vecinos.append(tuple(nuevo_estado))
    return vecinos

# ==========================================
# 1. BÚSQUEDA EN PROFUNDIDAD (DFS)
# ==========================================
def dfs(n):
    pila = [()]
    pasos = 0
    longitud_anterior = -1

    while pila:
        estado = pila.pop()
        pasos += 1

        if len(estado) > longitud_anterior:
            imprimir_paso(pasos, estado, n, "Constructivo")
            longitud_anterior = len(estado)

        if len(estado) == n: return estado

        for fila in reversed(range(n)):
            nuevo = estado + (fila,)
            if es_valido(nuevo): pila.append(nuevo)
    return None

# ==========================================
# 2. BÚSQUEDA EN ANCHURA (BFS)
# ==========================================
def bfs(n):
    cola = deque([()])
    pasos = 0
    longitud_anterior = -1

    while cola:
        estado = cola.popleft()
        pasos += 1

        if len(estado) > longitud_anterior:
            imprimir_paso(pasos, estado, n, "Constructivo")
            longitud_anterior = len(estado)

        if len(estado) == n:
            return estado

        for fila in range(n):
            nuevo = estado + (fila,)
            if es_valido(nuevo): cola.append(nuevo)
    return None

# ==========================================
# 3. PROFUNDIDAD ITERATIVA (IDDFS)
# ==========================================
def iddfs(n):
    pasos = 0
    maxima_profundidad_global = -1

    for limite in range(n + 1):
        print(f"\n>>> Iniciando búsqueda con Límite de Profundidad: {limite} <<<")
        pila = [()]

        while pila:
            estado = pila.pop()
            pasos += 1

            if len(estado) > maxima_profundidad_global:
                imprimir_paso(pasos, estado, n, "Constructivo")
                maxima_profundidad_global = len(estado)

            if len(estado) == n: return estado

            if len(estado) < limite:
                for fila in reversed(range(n)):
                    nuevo = estado + (fila,)
                    if es_valido(nuevo): pila.append(nuevo)
    return None

# ==========================================
# 4 y 5. VORAZ Y A* (A Star)
# ==========================================
def busqueda_heuristica(n, tipo="A_Star"):
    frontera = [(0, ())]
    pasos = 0
    longitud_anterior = -1

    while frontera:
        _, estado = heapq.heappop(frontera)
        pasos += 1

        if len(estado) > longitud_anterior:
            imprimir_paso(pasos, estado, n, "Constructivo")
            longitud_anterior = len(estado)

        if len(estado) == n: return estado

        for fila in range(n):
            nuevo = estado + (fila,)
            if es_valido(nuevo):
                g = len(nuevo)
                h = n - len(nuevo)

                if tipo == "Voraz":
                    f = h
                else:
                    f = g + h

                heapq.heappush(frontera, (f, nuevo))
    return None

# ==========================================
# 6. BÚSQUEDA TABÚ (Búsqueda Local)
# ==========================================
def tabu_search(n):
    estado_actual = tuple(random.randint(0, n-1) for _ in range(n))
    mejor_estado = estado_actual
    lista_tabu = deque(maxlen=5)

    mejor_conflictos_impreso = contar_conflictos(estado_actual)
    imprimir_paso(0, estado_actual, n, "Local")

    pasos = 0
    while contar_conflictos(mejor_estado) > 0 and pasos < 100:
        pasos += 1

        vecinos = generar_vecinos_locales(estado_actual)
        vecinos.sort(key=lambda x: contar_conflictos(x))

        for vecino in vecinos:
            if vecino not in lista_tabu:
                estado_actual = vecino
                lista_tabu.append(vecino)

                conflictos_actuales = contar_conflictos(estado_actual)
                if conflictos_actuales < contar_conflictos(mejor_estado):
                    mejor_estado = estado_actual

                if conflictos_actuales < mejor_conflictos_impreso:
                    imprimir_paso(pasos, estado_actual, n, "Local")
                    mejor_conflictos_impreso = conflictos_actuales
                break

    return mejor_estado if contar_conflictos(mejor_estado) == 0 else None

# ==========================================
# 7. RECOCIDO SIMULADO (Búsqueda Local)
# ==========================================
def recocido_simulado(n):
    estado_actual = tuple(random.randint(0, n-1) for _ in range(n))
    temperatura = 100.0
    tasa_enfriamiento = 0.95

    mejor_conflictos_impreso = contar_conflictos(estado_actual)
    imprimir_paso(0, estado_actual, n, "Local")

    pasos = 0
    while contar_conflictos(estado_actual) > 0 and temperatura > 0.1:
        pasos += 1

        col_random = random.randint(0, n-1)
        fila_random = random.randint(0, n-1)
        vecino = list(estado_actual)
        vecino[col_random] = fila_random
        vecino = tuple(vecino)

        delta = contar_conflictos(vecino) - contar_conflictos(estado_actual)

        if delta < 0 or random.random() < math.exp(-delta / temperatura):
            estado_actual = vecino

            conflictos_actuales = contar_conflictos(estado_actual)
            if conflictos_actuales < mejor_conflictos_impreso:
                imprimir_paso(pasos, estado_actual, n, "Local")
                mejor_conflictos_impreso = conflictos_actuales

        temperatura *= tasa_enfriamiento

    return estado_actual if contar_conflictos(estado_actual) == 0 else None

# ==========================================
# PROGRAMA PRINCIPAL Y MENÚ
# ==========================================
def main():
    print("\n========================================")
    print("  RESOLUTOR MULTI-ALGORITMO (N REINAS)  ")
    print("========================================")

    # Bucle para pedir y validar el tamaño del tablero
    while True:
        try:
            n = int(input("Ingrese el tamaño del tablero (un solo número para un tablero NxN con N reinas): "))
            if n < 4 and n != 1:
                print("Nota: El problema no tiene solución para tableros de 2x2 o 3x3. Por favor ingresa un número mayor o igual a 4.")
            elif n <= 0:
                print("Por favor, ingresa un número entero positivo.")
            else:
                break
        except ValueError:
            print("Entrada inválida. Por favor, ingresa un número entero.")

    while True:
        print("\n========================================")
        print(f"      MENÚ DE ALGORITMOS ({n} REINAS)      ")
        print("========================================")
        print("--- Algoritmos Constructivos ---")
        print("1. Búsqueda en Profundidad (DFS)")
        print("2. Búsqueda en Anchura (BFS)")
        print("3. Profundidad Iterativa (IDDFS)")
        print("4. Búsqueda Voraz (Greedy)")
        print("5. Búsqueda A* (A Star)")
        print("--- Algoritmos de Búsqueda Local ---")
        print("6. Búsqueda Tabú")
        print("7. Recocido Simulado")
        print("8. Cambiar tamaño del tablero")
        print("9. Salir")

        opcion = input("\nElige una opción (1-9): ").strip()

        if opcion == "9":
            print("Saliendo del programa...")
            break
        elif opcion == "8":
            # Permite al usuario regresar y cambiar el tamaño de N
            main()
            return

        solucion = None
        if opcion == "1": solucion = dfs(n)
        elif opcion == "2": solucion = bfs(n)
        elif opcion == "3": solucion = iddfs(n)
        elif opcion == "4": solucion = busqueda_heuristica(n, "Voraz")
        elif opcion == "5": solucion = busqueda_heuristica(n, "A_Star")
        elif opcion == "6": solucion = tabu_search(n)
        elif opcion == "7": solucion = recocido_simulado(n)
        else:
            print("Opción inválida.")
            continue

        print("\n========================================")

        # Evitar errores si la Búsqueda Local devuelve None
        if solucion is not None:
            print(" ¡SOLUCIÓN FINAL ENCONTRADA! ")
            print(f"Ruta: {solucion}")
            imprimir_tablero_simple(solucion, n)
        else:
            print("El algoritmo terminó sin encontrar la solución óptima (común en Búsqueda Local si se estanca).")

if __name__ == "__main__":
    main()