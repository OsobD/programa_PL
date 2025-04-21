from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, PULP_CBC_CMD, value
import numpy as np

def resolver_modelo_lineal(datos):
    """
    Resuelve un problema de programación lineal con los datos proporcionados.
    
    Args:
        datos: Diccionario con los siguientes campos:
            - num_variables: Número de variables de decisión (2-5)
            - num_restricciones: Número de restricciones (2-5)
            - coef_objetivo: Lista de coeficientes de la función objetivo
            - tipo_operacion: "maximizar" o "minimizar"
            - coef_restricciones: Lista de listas con los coeficientes de las restricciones
            - operadores: Lista con los operadores de las restricciones ("<=", ">=", "=")
            - lados_derechos: Lista con los valores de los lados derechos de las restricciones
    
    Returns:
        Diccionario con los resultados del modelo:
            - status: Estado de la solución
            - valor_objetivo: Valor óptimo de la función objetivo
            - variables: Valores de las variables de decisión
            - tiempo_ejecucion: Tiempo de ejecución del modelo
            - error: Mensaje de error (si ocurre)
    """
    try:
        # Crear el problema
        if datos["tipo_operacion"] == "maximizar":
            prob = LpProblem("Problema_PL", LpMaximize)
        else:
            prob = LpProblem("Problema_PL", LpMinimize)
        
        # Crear variables
        num_vars = datos["num_variables"]
        variables = [LpVariable(f"x{i}", lowBound=0) for i in range(1, num_vars + 1)]
        
        # Definir función objetivo
        coef_obj = datos["coef_objetivo"]
        prob += sum(coef_obj[i] * variables[i] for i in range(num_vars))
        
        # Añadir restricciones
        for i in range(datos["num_restricciones"]):
            coefs = datos["coef_restricciones"][i]
            operador = datos["operadores"][i]
            lado_derecho = datos["lados_derechos"][i]
            
            expresion = sum(coefs[j] * variables[j] for j in range(num_vars))
            
            if operador == "<=":
                prob += (expresion <= lado_derecho)
            elif operador == ">=":
                prob += (expresion >= lado_derecho)
            else:  # operador == "="
                prob += (expresion == lado_derecho)
        
        # Resolver el problema
        prob.solve(PULP_CBC_CMD(msg=False))
        
        # Preparar resultados
        resultados = {
            "status": prob.status,
            "status_text": prob.status == 1 and "Óptimo" or "No óptimo",
            "valor_objetivo": value(prob.objective),
            "variables": [{"nombre": f"x{i+1}", "valor": value(variables[i])} for i in range(num_vars)],
            "error": None
        }
        
        return resultados
    
    except Exception as e:
        return {
            "status": -1,
            "status_text": "Error",
            "valor_objetivo": None,
            "variables": None,
            "error": str(e)
        }
