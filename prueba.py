# Importamos la biblioteca PuLP para programación lineal
from pulp import *

# Creamos el problema de maximización
prob = LpProblem("Maximización_de_Ganancias_Omega", LpMaximize)

# Definimos las variables de decisión (productos 1, 2 y 3)
x1 = LpVariable("Producto_1", lowBound=0, cat='Integer')  # Cantidad del producto 1
x2 = LpVariable("Producto_2", lowBound=0, cat='Integer')  # Cantidad del producto 2
x3 = LpVariable("Producto_3", lowBound=0, cat='Integer')  # Cantidad del producto 3

# Función objetivo: Maximizar ganancias
prob += 50*x1 + 20*x2 + 25*x3, "Ganancia Total"

# Restricciones de capacidad de máquinas (en horas-máquina por semana)
# Fresadora: 9 horas para P1, 3 horas para P2 y 5 horas para P3, con 500 horas disponibles
prob += 9*x1 + 3*x2 + 5*x3 <= 500, "Restricción de Fresadora"

# Torno: 5 horas para P1, 4 horas para P2 y 0 horas para P3, con 350 horas disponibles
prob += 5*x1 + 4*x2 + 0*x3 <= 350, "Restricción de Torno"

# Rectificadora: 3 horas para P1, 0 horas para P2 y 3 horas para P3, con 150 horas disponibles
prob += 3*x1 + 0*x2 + 3*x3 <= 150, "Restricción de Rectificadora"

# Restricción de demanda máxima para el producto 3 (20 unidades por semana)
prob += x3 <= 20, "Restricción de demanda del Producto 3"

# Resolvemos el problema
prob.solve()

# Imprimimos el estado de la solución
print(f"Estado: {LpStatus[prob.status]}")

# Imprimimos los resultados
print("\nResultados óptimos:")
print(f"Producto 1: {value(x1)} unidades")
print(f"Producto 2: {value(x2)} unidades")
print(f"Producto 3: {value(x3)} unidades")
print(f"\nGanancia máxima: ${value(prob.objective)}")

# Analizamos el uso de recursos
print("\nAnálisis de restricciones:")
fresa_usado = 9*value(x1) + 3*value(x2) + 5*value(x3)
torno_usado = 5*value(x1) + 4*value(x2) + 0*value(x3)
rect_usado = 3*value(x1) + 0*value(x2) + 3*value(x3)

print(f"Fresadora: {fresa_usado}/500 horas ({fresa_usado/500*100:.1f}%)")
print(f"Torno: {torno_usado}/350 horas ({torno_usado/350*100:.1f}%)")
print(f"Rectificadora: {rect_usado}/150 horas ({rect_usado/150*100:.1f}%)")