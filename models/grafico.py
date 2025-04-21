import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Para usar matplotlib sin interfaz gráfica
import io
import base64
from itertools import combinations
from matplotlib.patches import Polygon

def calcular_interseccion(a1, b1, c1, a2, b2, c2):
    """
    Calcula el punto de intersección entre dos líneas:
    a1*x + b1*y = c1
    a2*x + b2*y = c2
    
    Retorna (x, y) o None si son paralelas
    """
    # Verificar si son paralelas
    det = a1*b2 - a2*b1
    if abs(det) < 1e-10:  # Prácticamente paralelas
        return None
    
    # Calcular el punto de intersección
    x = (c1*b2 - c2*b1) / det
    y = (a1*c2 - a2*c1) / det
    
    return (x, y)

def evaluar_restriccion(punto, a, b, c, operador):
    """
    Evalúa si un punto (x, y) satisface una restricción a*x + b*y operador c
    Retorna True si satisface, False en caso contrario
    """
    x, y = punto
    valor = a*x + b*y
    
    if operador == "<=":
        return valor <= c + 1e-10  # Pequeña tolerancia numérica
    elif operador == ">=":
        return valor >= c - 1e-10  # Pequeña tolerancia numérica
    else:  # operador == "="
        return abs(valor - c) < 1e-10

def generar_metodo_grafico(datos):
    """
    Genera una visualización gráfica de un problema de programación lineal con 2 variables.
    
    Args:
        datos: Diccionario con los datos del problema:
            - coef_objetivo: Lista de coeficientes de la función objetivo [c1, c2]
            - tipo_operacion: "maximizar" o "minimizar"
            - coef_restricciones: Lista de listas con los coeficientes de las restricciones [[a1, b1], [a2, b2], ...]
            - operadores: Lista de operadores ("<=", ">=", "=")
            - lados_derechos: Lista de valores del lado derecho [c1, c2, ...]
    
    Returns:
        Un diccionario con:
            - imagen_base64: La imagen en formato base64
            - puntos_esquina: Lista de puntos en las esquinas de la región factible
            - punto_optimo: El punto óptimo (x*, y*)
            - valor_optimo: El valor óptimo de la función objetivo
            - error: Mensaje de error (si ocurre)
    """
    try:
        # Información de diagnóstico
        print("Datos recibidos en generar_metodo_grafico:")
        print(f"Coeficientes objetivo: {datos['coef_objetivo']}")
        print(f"Tipo operación: {datos['tipo_operacion']}")
        
        if len(datos['coef_objetivo']) != 2:
            return {"error": "El método gráfico solo funciona para problemas con 2 variables"}
        
        # Configurar la figura
        plt.figure(figsize=(10, 8))
        
        # Definir límites iniciales para los ejes
        max_limit = 10
        min_limit = -1
        
        # Extraer coeficientes
        c1, c2 = datos['coef_objetivo']
        print(f"c1: {c1}, c2: {c2}")
        restricciones = []
        
        for i in range(datos['num_restricciones']):
            a, b = datos['coef_restricciones'][i]
            c = datos['lados_derechos'][i]
            op = datos['operadores'][i]
            restricciones.append((a, b, c, op))
        
        # Añadir restricciones de no negatividad si no están explícitas
        if all(r[0] != 1 or r[1] != 0 or r[3] != ">=" for r in restricciones):
            restricciones.append((1, 0, 0, ">="))  # x >= 0
        
        if all(r[0] != 0 or r[1] != 1 or r[3] != ">=" for r in restricciones):
            restricciones.append((0, 1, 0, ">="))  # y >= 0
        
        # Buscar todos los puntos de intersección
        puntos_interseccion = []
        
        # Agregar el origen si es factible
        origen = (0, 0)
        if all(evaluar_restriccion(origen, *rest) for rest in restricciones):
            puntos_interseccion.append(origen)
            print("El origen (0,0) es factible")
        else:
            print("El origen (0,0) NO es factible")
        
        # Intersecciones en los ejes
        for a, b, c, op in restricciones:
            if abs(a) > 1e-10:  # Si a != 0
                punto_x = (c/a, 0)
                if all(evaluar_restriccion(punto_x, *rest) for rest in restricciones):
                    puntos_interseccion.append(punto_x)
            
            if abs(b) > 1e-10:  # Si b != 0
                punto_y = (0, c/b)
                if all(evaluar_restriccion(punto_y, *rest) for rest in restricciones):
                    puntos_interseccion.append(punto_y)
        
        # Intersecciones entre restricciones
        for (a1, b1, c1, op1), (a2, b2, c2, op2) in combinations(restricciones, 2):
            punto = calcular_interseccion(a1, b1, c1, a2, b2, c2)
            if punto and all(evaluar_restriccion(punto, *rest) for rest in restricciones):
                puntos_interseccion.append(punto)
        
        # Eliminar duplicados y puntos muy cercanos
        puntos_esquina = []
        for punto in puntos_interseccion:
            if not any(np.linalg.norm(np.array(punto) - np.array(p)) < 1e-8 for p in puntos_esquina):
                # Verificar que el punto sea no negativo
                x, y = punto
                if x >= 0 and y >= 0:
                    puntos_esquina.append(punto)
        
        print(f"Puntos de esquina encontrados: {puntos_esquina}")
        
        # Actualizar límites para los ejes
        if puntos_esquina:
            for x, y in puntos_esquina:
                max_limit = max(max_limit, x*1.2, y*1.2)
        
        # Configurar límites de los ejes
        plt.xlim(min_limit, max_limit)
        plt.ylim(min_limit, max_limit)
        
        # Graficar cada restricción
        x = np.linspace(min_limit, max_limit, 1000)
        
        for i, (a, b, c, op) in enumerate(restricciones):
            label = f"{a}x + {b}y {op} {c}"
            
            if abs(b) < 1e-10:  # Línea vertical b=0
                if a != 0:
                    plt.axvline(x=c/a, label=label, linestyle='--')
            else:
                y = (c - a*x) / b
                plt.plot(x, y, label=label, linestyle='--')
        
        # Sombrear la región factible si hay puntos factibles
        if len(puntos_esquina) >= 3:
            # Ordenar los puntos para formar un polígono convexo
            def ordenar_puntos_convexos(puntos):
                # Calcular el centroide
                centro = np.mean(puntos, axis=0)
                # Ordenar según el ángulo
                return sorted(puntos, key=lambda p: np.arctan2(p[1]-centro[1], p[0]-centro[0]))
            
            puntos_ordenados = ordenar_puntos_convexos(puntos_esquina)
            poligono = Polygon(puntos_ordenados, alpha=0.2, color='green', label='Región Factible')
            plt.gca().add_patch(poligono)
        
        # Encontrar el punto óptimo
        punto_optimo = None
        valor_optimo = float('-inf') if datos['tipo_operacion'] == 'maximizar' else float('inf')
        
        print("Evaluando puntos de esquina:")
        if puntos_esquina:
            for punto in puntos_esquina:
                x, y = punto
                # Asegurarse de usar correctamente los coeficientes de la función objetivo
                z = float(c1*x + c2*y)
                print(f"Punto ({x}, {y}): Valor F.O. = {z}")
                
                if datos['tipo_operacion'] == 'maximizar':
                    if z > valor_optimo or (abs(z - valor_optimo) < 1e-10 and np.linalg.norm(punto) < np.linalg.norm(punto_optimo or [float('inf'), float('inf')])):
                        valor_optimo = z
                        punto_optimo = punto
                else:  # minimizar
                    if z < valor_optimo or (abs(z - valor_optimo) < 1e-10 and np.linalg.norm(punto) < np.linalg.norm(punto_optimo or [float('inf'), float('inf')])):
                        valor_optimo = z
                        punto_optimo = punto
        
        if punto_optimo:
            print(f"Punto óptimo seleccionado: {punto_optimo} con valor {valor_optimo}")
        
        # Graficar los puntos esquina
        for i, (x, y) in enumerate(puntos_esquina):
            plt.plot(x, y, 'o', color='blue')
            plt.annotate(f'P{i+1}({x:.2f}, {y:.2f})', (x, y), 
                         textcoords="offset points", xytext=(0,10), ha='center')
        
        # Destacar el punto óptimo
        if punto_optimo:
            x_opt, y_opt = punto_optimo
            plt.plot(x_opt, y_opt, 'o', color='red', markersize=10)
            plt.annotate(f'Óptimo ({x_opt:.2f}, {y_opt:.2f})', 
                         (x_opt, y_opt), 
                         textcoords="offset points", 
                         xytext=(0,15), 
                         ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        # Graficar algunas líneas de nivel de la función objetivo
        if punto_optimo:
            z_vals = np.linspace(valor_optimo * 0.5, valor_optimo * 1.5, 5)
            
            for z in z_vals:
                if abs(c2) > 1e-10:  # Si c2 no es cero
                    y_obj = (z - c1*x) / c2
                    plt.plot(x, y_obj, 'g-', alpha=0.3, linewidth=1)
            
            # Destacar el nivel óptimo
            if abs(c2) > 1e-10:  # Si c2 no es cero
                y_opt_line = (valor_optimo - c1*x) / c2
                plt.plot(x, y_opt_line, 'g-', alpha=0.7, linewidth=2,
                        label=f'F.O. = {valor_optimo:.2f}')
        
        # Configurar etiquetas y leyenda
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.grid(True)
        plt.title(f"Método Gráfico - {datos['tipo_operacion'].capitalize()}: {c1}x₁ + {c2}x₂")
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        
        # Convertir la figura a una imagen en base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Preparar resultados para retornar
        resultado = {
            "imagen_base64": img_base64,
            "puntos_esquina": [],
            "punto_optimo": {"x": float(punto_optimo[0]), "y": float(punto_optimo[1])} if punto_optimo else None,
            "valor_optimo": float(valor_optimo) if punto_optimo is not None else None,
            "error": None
        }
        
        # Asegurarse de que los valores de la función objetivo son correctos para cada punto
        for x, y in puntos_esquina:
            valor_punto = float(c1 * x + c2 * y)
            resultado["puntos_esquina"].append({
                "x": float(x), 
                "y": float(y), 
                "valor": valor_punto
            })
        
        return resultado
    
    except Exception as e:
        import traceback
        return {
            "error": f"Error al generar el método gráfico: {str(e)}",
            "traceback": traceback.format_exc()
        }
