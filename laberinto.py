import random  

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

def mover(posicion, movimiento):
    x, y = posicion

    if movimiento == "arriba":
        return (x - 1, y)
    elif movimiento == "abajo":
        return (x + 1, y)
    elif movimiento == "izquierda":
        return (x, y - 1)
    elif movimiento == "derecha":
        return (x, y + 1)
    else:
        return posicion

def es_valido(tablero, posicion):
    m = len(tablero)
    n = len(tablero[0])
    x, y = posicion

    if 0 <= x < m and 0 <= y < n:
        if tablero[x][y] != "X":
            return True
    return False

#PROGRAMA PRINCIPAL 

m = int(input("Ingrese número de filas (m): "))
n = int(input("Ingrese número de columnas (n): "))
porcentaje = float(input("Ingrese porcentaje de obstáculos: "))

tablero = crear_tablero(m, n)
generar_obstaculos(tablero, porcentaje)

inicio = (0, 0)
meta = (m - 1, n - 1)

tablero[inicio[0]][inicio[1]] = "I"
tablero[meta[0]][meta[1]] = "M"

posicion_actual = inicio

print("\nTablero generado:")
imprimir_tablero(tablero)

movimientos = input("Ingrese lista de movimientos separados por coma: ")
movimientos = movimientos.split(",")

for mov in movimientos:
    mov = mov.strip().lower()
    nueva_posicion = mover(posicion_actual, mov)

    if es_valido(tablero, nueva_posicion):
        posicion_actual = nueva_posicion
        print(f"Movimiento válido: {mov}")
    else:
        print(f"Movimiento inválido: {mov}")

    if posicion_actual == meta:
        print("¡Meta alcanzada!")
        break

print("\nPosición final:", posicion_actual)