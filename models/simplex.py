import numpy as np

def resolver_simplex_paso_a_paso(datos):
    """
    Resuelve un problema de programación lineal usando el método Simplex paso a paso.
    
    Args:
        datos: Diccionario con los siguientes campos:
            - num_variables: Número de variables de decisión
            - num_restricciones: Número de restricciones
            - coef_objetivo: Lista de coeficientes de la función objetivo
            - tipo_operacion: "maximizar" o "minimizar"
            - coef_restricciones: Lista de listas con los coeficientes de las restricciones
            - operadores: Lista con los operadores de las restricciones ("<=", ">=", "=")
            - lados_derechos: Lista con los valores de los lados derechos de las restricciones
    
    Returns:
        Diccionario con los resultados y pasos del método Simplex:
            - pasos: Lista de pasos con las tablas intermedias
            - resultado_final: Resultado final del problema
            - metodo: "simplex" o "gran_m" según el método utilizado
    """
    # Determinar si se necesita el método de la Gran M
    necesita_gran_m = any(op in [">=", "="] for op in datos["operadores"])
    
    if necesita_gran_m:
        return metodo_gran_m(datos)
    else:
        return metodo_simplex_estandar(datos)

def metodo_simplex_estandar(datos):
    """
    Aplica el método Simplex estándar para problemas de maximización con restricciones <=
    """
    num_vars = datos["num_variables"]
    num_rest = datos["num_restricciones"]
    coef_obj = datos["coef_objetivo"].copy()
    coef_rest = [row.copy() for row in datos["coef_restricciones"]]
    lados_derechos = datos["lados_derechos"].copy()
    tipo_operacion = datos["tipo_operacion"]
    
    # Para minimización, cambiamos el signo de la función objetivo
    es_minimizacion = tipo_operacion == "minimizar"
    if es_minimizacion:
        coef_obj = [-c for c in coef_obj]
    
    # Número de variables de holgura (una por cada restricción <=)
    num_vars_holgura = num_rest
    
    # Crear la matriz inicial del tableau
    # Columnas: Z, X1...Xn, S1...Sm, Sol
    num_cols = 1 + num_vars + num_vars_holgura + 1
    num_filas = 1 + num_rest  # Fila objetivo + filas de restricciones
    
    tableau = np.zeros((num_filas, num_cols))
    
    # Llenar la fila objetivo (fila 0)
    tableau[0, 0] = 1  # Coeficiente de Z
    for j in range(num_vars):
        tableau[0, j+1] = -coef_obj[j]  # Coeficientes de X con signo negativo
    
    # Llenar las filas de restricciones
    for i in range(num_rest):
        # Coeficientes de las variables originales
        for j in range(num_vars):
            tableau[i+1, j+1] = coef_rest[i][j]
        
        # Variable de holgura (1 en la posición correspondiente)
        tableau[i+1, num_vars+i+1] = 1
        
        # Lado derecho
        tableau[i+1, -1] = lados_derechos[i]
    
    # Lista para almacenar cada paso
    pasos = []
    
    # Añadir tableau inicial
    nombres_columnas = ["Z"] + [f"X{j+1}" for j in range(num_vars)] + [f"S{j+1}" for j in range(num_vars_holgura)] + ["Sol"]
    nombres_filas = ["f1"] + [f"f{i+2}" for i in range(num_rest)]
    
    pasos.append({
        "paso": 0,
        "descripcion": "Tableau inicial",
        "tableau": tableau.copy(),
        "nombres_columnas": nombres_columnas,
        "nombres_filas": nombres_filas
    })
    
    # Iteraciones del método Simplex
    iteracion = 1
    max_iteraciones = 20  # Evitar bucles infinitos
    
    while iteracion <= max_iteraciones:
        # Verificar si ya se alcanzó la solución óptima 
        # Para maximización: todos los coeficientes deben ser >= 0
        # Para minimización: como invertimos los signos, la condición es la misma
        if all(tableau[0, 1:num_vars+num_vars_holgura+1] >= 0):
            break
        
        # Encontrar la columna pivote:
        # Para maximización: el valor más negativo en la fila objetivo
        # Para minimización: como invertimos los signos, la lógica es la misma
        col_pivote = np.argmin(tableau[0, 1:num_vars+num_vars_holgura+1]) + 1
        
        # Calcular los cocientes para identificar la fila pivote
        cocientes = []
        for i in range(1, num_filas):
            if tableau[i, col_pivote] <= 0:
                cocientes.append(float('inf'))
            else:
                cocientes.append(tableau[i, -1] / tableau[i, col_pivote])
        
        if all(c == float('inf') for c in cocientes):
            return {
                "pasos": pasos,
                "metodo": "simplex",
                "resultado_final": {
                    "status_text": "Problema no acotado",
                    "valor_objetivo": None,
                    "variables": None
                }
            }
        
        # Encontrar la fila pivote (el menor cociente positivo)
        fila_pivote = cocientes.index(min(cocientes)) + 1
        
        # Registrar la selección de pivote
        pasos.append({
            "paso": iteracion,
            "descripcion": f"Selección de pivote",
            "columna_pivote": col_pivote,
            "columna_pivote_nombre": nombres_columnas[col_pivote],
            "fila_pivote": fila_pivote,
            "fila_pivote_nombre": nombres_filas[fila_pivote-1],
            "valor_pivote": tableau[fila_pivote, col_pivote],
            "cocientes": cocientes
        })
        
        # Normalizar la fila pivote
        valor_pivote = tableau[fila_pivote, col_pivote]
        tableau[fila_pivote] = tableau[fila_pivote] / valor_pivote
        
        # Registrar la normalización
        pasos.append({
            "paso": iteracion,
            "descripcion": f"Normalización de la fila pivote",
            "operacion": f"{nombres_filas[fila_pivote-1]} = {nombres_filas[fila_pivote-1]} / {valor_pivote:.4f}",
            "tableau": tableau.copy(),
            "nombres_columnas": nombres_columnas,
            "nombres_filas": nombres_filas
        })
        
        # Hacer ceros en la columna pivote
        for i in range(num_filas):
            if i != fila_pivote:
                factor = tableau[i, col_pivote]
                tableau[i] = tableau[i] - factor * tableau[fila_pivote]
                
                # Registrar cada operación de fila
                if abs(factor) > 1e-10:  # Solo registrar si el factor no es prácticamente cero
                    pasos.append({
                        "paso": iteracion,
                        "descripcion": "Operación de fila",
                        "operacion": f"{nombres_filas[i]} = {nombres_filas[i]} - {factor:.4f} * {nombres_filas[fila_pivote-1]}",
                        "tableau": tableau.copy(),
                        "nombres_columnas": nombres_columnas,
                        "nombres_filas": nombres_filas
                    })
        
        iteracion += 1
    
    # Extraer la solución final
    # Identificar variables básicas (columnas con exactamente un 1 y el resto ceros)
    variables_basicas = []
    valores_basicos = []
    
    for j in range(1, num_vars + num_vars_holgura + 1):
        col = tableau[:, j]
        if (col == 1).sum() == 1 and (col == 0).sum() == num_filas - 1:
            # Es una variable básica
            fila = np.where(col == 1)[0][0]
            variables_basicas.append(j)
            valores_basicos.append(tableau[fila, -1])
        else:
            variables_basicas.append(j)
            valores_basicos.append(0)
    
    # Preparar el resultado final
    resultado = {
        "status_text": "Óptimo" if iteracion <= max_iteraciones else "Número máximo de iteraciones alcanzado",
        "valor_objetivo": tableau[0, -1] if not es_minimizacion else -tableau[0, -1],
        "variables": []
    }
    
    # Extraer los valores de las variables originales
    for j in range(num_vars):
        idx = j + 1  # Índice en el tableau
        valor = 0
        
        if idx in variables_basicas:
            pos = variables_basicas.index(idx)
            valor = valores_basicos[pos]
        
        resultado["variables"].append({
            "nombre": f"x{j+1}",
            "valor": valor
        })
    
    return {
        "pasos": pasos,
        "metodo": "simplex",
        "resultado_final": resultado
    }

def metodo_gran_m(datos):
    """
    Aplica el método de la Gran M para problemas con restricciones mixtas
    """
    num_vars = datos["num_variables"]
    num_rest = datos["num_restricciones"]
    coef_obj = datos["coef_objetivo"].copy()
    coef_rest = [row.copy() for row in datos["coef_restricciones"]]
    operadores = datos["operadores"].copy()
    lados_derechos = datos["lados_derechos"].copy()
    tipo_operacion = datos["tipo_operacion"]
    
    # Para minimización, cambiamos el signo de la función objetivo
    es_minimizacion = tipo_operacion == "minimizar"
    if es_minimizacion:
        coef_obj = [-c for c in coef_obj]
    
    # Asegurarse de que todos los lados derechos sean no negativos
    for i in range(num_rest):
        if lados_derechos[i] < 0:
            lados_derechos[i] *= -1
            coef_rest[i] = [-c for c in coef_rest[i]]
            if operadores[i] == "<=":
                operadores[i] = ">="
            elif operadores[i] == ">=":
                operadores[i] = "<="
    
    # Contar variables adicionales necesarias
    num_vars_holgura = sum(1 for op in operadores if op in ["<=", ">="]) 
    num_vars_artificiales = sum(1 for op in operadores if op in [">=", "="])
    
    # Mapeo de las variables adicionales para cada restricción
    # (tipo, índice) donde tipo puede ser 'h' (holgura) o 'a' (artificial)
    vars_adicionales = []
    idx_holgura = 0
    idx_artificial = 0
    
    for op in operadores:
        if op == "<=":
            vars_adicionales.append([('h', idx_holgura)])
            idx_holgura += 1
        elif op == ">=":
            vars_adicionales.append([('h', idx_holgura), ('a', idx_artificial)])
            idx_holgura += 1
            idx_artificial += 1
        else:  # op == "="
            vars_adicionales.append([('a', idx_artificial)])
            idx_artificial += 1
    
    # Crear la matriz inicial del tableau
    # Columnas: Z, X1...Xn, S1...Sm, R1...Rk, Sol
    num_cols = 1 + num_vars + num_vars_holgura + num_vars_artificiales + 1
    num_filas = 1 + num_rest  # Fila objetivo + filas de restricciones
    
    # Crearemos dos matrices: una para los coeficientes numéricos y otra para los coeficientes simbólicos M
    tableau_numerico = np.zeros((num_filas, num_cols))
    tableau_M = np.zeros((num_filas, num_cols))  # Para coeficientes de M
    
    # Llenar la fila objetivo (fila 0)
    tableau_numerico[0, 0] = 1  # Coeficiente de Z
    for j in range(num_vars):
        tableau_numerico[0, j+1] = -coef_obj[j]  # Coeficientes de X con signo negativo
    
    # Añadir coeficientes M para variables artificiales
    for i, vars_lista in enumerate(vars_adicionales):
        for tipo, idx in vars_lista:
            if tipo == 'a':  # Variable artificial
                # Para maximización y minimización (como ya invertimos la FO), siempre restamos M
                col_idx = 1 + num_vars + num_vars_holgura + idx
                tableau_M[0, col_idx] = -1  # -M
    
    # Llenar las filas de restricciones
    for i in range(num_rest):
        # Coeficientes de las variables originales
        for j in range(num_vars):
            tableau_numerico[i+1, j+1] = coef_rest[i][j]
        
        # Variables adicionales
        for tipo, idx in vars_adicionales[i]:
            if tipo == 'h':  # Variable de holgura
                col_idx = 1 + num_vars + idx
                # Si es <=, se suma; si es >=, se resta
                tableau_numerico[i+1, col_idx] = 1 if operadores[i] == "<=" else -1
            else:  # Variable artificial
                col_idx = 1 + num_vars + num_vars_holgura + idx
                tableau_numerico[i+1, col_idx] = 1
        
        # Lado derecho
        tableau_numerico[i+1, -1] = lados_derechos[i]
    
    # Crear nombres para las columnas y filas
    nombres_columnas = ["Z"] + [f"X{j+1}" for j in range(num_vars)]
    nombres_columnas += [f"S{j+1}" for j in range(num_vars_holgura)]
    nombres_columnas += [f"R{j+1}" for j in range(num_vars_artificiales)]
    nombres_columnas += ["Sol"]
    
    nombres_filas = ["f1"] + [f"f{i+2}" for i in range(num_rest)]
    
    # Lista para almacenar cada paso
    pasos = []
    
    # Combinar los coeficientes numéricos con los simbólicos M para mostrar
    tableau_combinado = {}
    for i in range(num_filas):
        for j in range(num_cols):
            if i not in tableau_combinado:
                tableau_combinado[i] = {}
            
            if tableau_M[i][j] != 0:
                tableau_combinado[i][j] = {
                    "numerico": tableau_numerico[i][j],
                    "simbolico": tableau_M[i][j],
                    "tiene_M": True
                }
            else:
                tableau_combinado[i][j] = {
                    "numerico": tableau_numerico[i][j],
                    "simbolico": 0,
                    "tiene_M": False
                }
    
    # Añadir tableau inicial
    pasos.append({
        "paso": 0,
        "descripcion": "Tableau inicial",
        "tableau_numerico": tableau_numerico.copy(),
        "tableau_M": tableau_M.copy(),
        "tableau_combinado": tableau_combinado.copy(),
        "nombres_columnas": nombres_columnas,
        "nombres_filas": nombres_filas
    })
    
    # Si hay variables artificiales, hay que hacer ceros en la fila objetivo
    if num_vars_artificiales > 0:
        # Para cada variable artificial, ajustar la fila objetivo
        for i in range(num_rest):
            for tipo, idx in vars_adicionales[i]:
                if tipo == 'a':  # Es una variable artificial
                    col_idx = 1 + num_vars + num_vars_holgura + idx
                    if tableau_M[0, col_idx] != 0:  # Si tiene coeficiente M
                        # Multiplicar la fila de restricción por el coeficiente de M y restar de la F.O.
                        m_coef = tableau_M[0, col_idx]  # Coeficiente de M (normalmente -1)
                        
                        # Actualizar coeficientes numéricos
                        tableau_numerico[0] = tableau_numerico[0] - m_coef * tableau_numerico[i+1]
                        
                        # Actualizar coeficientes simbólicos M
                        tableau_M[0] = tableau_M[0] - m_coef * tableau_M[i+1]
                        
                        # Actualizar tableau combinado
                        nuevo_combinado = {}
                        for ii in range(num_filas):
                            for jj in range(num_cols):
                                if ii not in nuevo_combinado:
                                    nuevo_combinado[ii] = {}
                                
                                if tableau_M[ii][jj] != 0:
                                    nuevo_combinado[ii][jj] = {
                                        "numerico": tableau_numerico[ii][jj],
                                        "simbolico": tableau_M[ii][jj],
                                        "tiene_M": True
                                    }
                                else:
                                    nuevo_combinado[ii][jj] = {
                                        "numerico": tableau_numerico[ii][jj],
                                        "simbolico": 0,
                                        "tiene_M": False
                                    }
                        
                        # Registrar la operación
                        pasos.append({
                            "paso": 0,
                            "descripcion": "Ajuste de fila objetivo para variables artificiales",
                            "operacion": f"{nombres_filas[0]} = {nombres_filas[0]} - {m_coef:.0f}M * {nombres_filas[i+1]}",
                            "tableau_numerico": tableau_numerico.copy(),
                            "tableau_M": tableau_M.copy(),
                            "tableau_combinado": nuevo_combinado,
                            "nombres_columnas": nombres_columnas,
                            "nombres_filas": nombres_filas
                        })
    
    # Iteraciones del método Simplex
    iteracion = 1
    max_iteraciones = 20  # Evitar bucles infinitos
    
    while iteracion <= max_iteraciones:
        # Verificar si ya se alcanzó la solución óptima
        # Para el criterio de optimalidad, primero verificamos los coeficientes con M
        hay_negativos_M = any(tableau_M[0, j] < 0 for j in range(1, num_cols-1))
        
        # Si no hay coeficientes M negativos, verificamos los numéricos
        if not hay_negativos_M:
            if all(tableau_numerico[0, j] >= 0 for j in range(1, num_cols-1)):
                break
        
        # Encontrar la columna pivote
        # Prioridad a los coeficientes con M negativos, luego a los numéricos
        col_pivote = 0
        valor_minimo = 0
        
        for j in range(1, num_cols-1):
            # Si el coeficiente M es negativo, es mayor prioridad
            if tableau_M[0, j] < 0:
                if tableau_M[0, j] < valor_minimo:
                    valor_minimo = tableau_M[0, j]
                    col_pivote = j
            # Si no hay M negativos o están empatados, usamos los numéricos
            elif tableau_M[0, j] == 0 and tableau_numerico[0, j] < 0:
                if col_pivote == 0 or tableau_numerico[0, j] < tableau_numerico[0, col_pivote]:
                    col_pivote = j
        
        # Si no se encontró columna pivote, verificar si es por error numérico
        if col_pivote == 0:
            break
        
        # Calcular los cocientes para identificar la fila pivote
        cocientes = []
        for i in range(1, num_filas):
            if tableau_numerico[i, col_pivote] <= 0:
                cocientes.append(float('inf'))
            else:
                cocientes.append(tableau_numerico[i, -1] / tableau_numerico[i, col_pivote])
        
        if all(c == float('inf') for c in cocientes):
            return {
                "pasos": pasos,
                "metodo": "gran_m",
                "resultado_final": {
                    "status_text": "Problema no acotado",
                    "valor_objetivo": None,
                    "variables": None
                }
            }
        
        # Encontrar la fila pivote (el menor cociente positivo)
        fila_pivote = cocientes.index(min(cocientes)) + 1
        
        # Valor pivote (siempre es numérico, no tiene M)
        valor_pivote = tableau_numerico[fila_pivote, col_pivote]
        
        # Registrar la selección de pivote
        pasos.append({
            "paso": iteracion,
            "descripcion": f"Selección de pivote",
            "columna_pivote": col_pivote,
            "columna_pivote_nombre": nombres_columnas[col_pivote],
            "fila_pivote": fila_pivote,
            "fila_pivote_nombre": nombres_filas[fila_pivote-1],
            "valor_pivote": valor_pivote,
            "cocientes": cocientes,
            "tableau_numerico": tableau_numerico.copy(),
            "tableau_M": tableau_M.copy(),
            "nombres_columnas": nombres_columnas,
            "nombres_filas": nombres_filas
        })
        
        # Normalizar la fila pivote
        tableau_numerico[fila_pivote] = tableau_numerico[fila_pivote] / valor_pivote
        tableau_M[fila_pivote] = tableau_M[fila_pivote] / valor_pivote
        
        # Actualizar tableau combinado
        nuevo_combinado = {}
        for i in range(num_filas):
            for j in range(num_cols):
                if i not in nuevo_combinado:
                    nuevo_combinado[i] = {}
                
                if tableau_M[i][j] != 0:
                    nuevo_combinado[i][j] = {
                        "numerico": tableau_numerico[i][j],
                        "simbolico": tableau_M[i][j],
                        "tiene_M": True
                    }
                else:
                    nuevo_combinado[i][j] = {
                        "numerico": tableau_numerico[i][j],
                        "simbolico": 0,
                        "tiene_M": False
                    }
        
        # Registrar la normalización
        pasos.append({
            "paso": iteracion,
            "descripcion": f"Normalización de la fila pivote",
            "operacion": f"{nombres_filas[fila_pivote-1]} = {nombres_filas[fila_pivote-1]} / {valor_pivote:.4f}",
            "tableau_numerico": tableau_numerico.copy(),
            "tableau_M": tableau_M.copy(),
            "tableau_combinado": nuevo_combinado,
            "nombres_columnas": nombres_columnas,
            "nombres_filas": nombres_filas
        })
        
        # Hacer ceros en la columna pivote
        for i in range(num_filas):
            if i != fila_pivote:
                # Factor numérico
                factor_numerico = tableau_numerico[i, col_pivote]
                # Factor para términos con M
                factor_M = tableau_M[i, col_pivote]
                
                # Ajustar coeficientes numéricos
                tableau_numerico[i] = tableau_numerico[i] - factor_numerico * tableau_numerico[fila_pivote]
                # Ajustar coeficientes con M
                tableau_M[i] = tableau_M[i] - factor_M * tableau_M[fila_pivote] - factor_numerico * tableau_M[fila_pivote]
                
                # Actualizar tableau combinado
                nuevo_combinado = {}
                for ii in range(num_filas):
                    for jj in range(num_cols):
                        if ii not in nuevo_combinado:
                            nuevo_combinado[ii] = {}
                        
                        if tableau_M[ii][jj] != 0:
                            nuevo_combinado[ii][jj] = {
                                "numerico": tableau_numerico[ii][jj],
                                "simbolico": tableau_M[ii][jj],
                                "tiene_M": True
                            }
                        else:
                            nuevo_combinado[ii][jj] = {
                                "numerico": tableau_numerico[ii][jj],
                                "simbolico": 0,
                                "tiene_M": False
                            }
                
                # Determinar la operación a mostrar
                if abs(factor_numerico) > 1e-10 or abs(factor_M) > 1e-10:
                    operacion = f"{nombres_filas[i]} = {nombres_filas[i]}"
                    if abs(factor_numerico) > 1e-10:
                        operacion += f" - {factor_numerico:.4f} * {nombres_filas[fila_pivote-1]}"
                    if abs(factor_M) > 1e-10:
                        if abs(factor_numerico) > 1e-10:
                            operacion += " -"
                        else:
                            operacion += " -"
                        operacion += f" {abs(factor_M):.0f}M * {nombres_filas[fila_pivote-1]}"
                    
                    # Registrar cada operación de fila
                    pasos.append({
                        "paso": iteracion,
                        "descripcion": "Operación de fila",
                        "operacion": operacion,
                        "tableau_numerico": tableau_numerico.copy(),
                        "tableau_M": tableau_M.copy(),
                        "tableau_combinado": nuevo_combinado,
                        "nombres_columnas": nombres_columnas,
                        "nombres_filas": nombres_filas
                    })
        
        iteracion += 1
    
    # Verificar si hay variables artificiales en la base
    for j in range(1 + num_vars + num_vars_holgura, 1 + num_vars + num_vars_holgura + num_vars_artificiales):
        col_num = tableau_numerico[:, j]
        col_M = tableau_M[:, j]
        
        for i in range(1, num_filas):
            # Si hay un valor significativo en una variable artificial y el lado derecho no es cero
            if (abs(col_num[i]) > 1e-10 or abs(col_M[i]) > 1e-10) and abs(tableau_numerico[i, -1]) > 1e-10:
                return {
                    "pasos": pasos,
                    "metodo": "gran_m",
                    "resultado_final": {
                        "status_text": "Problema sin solución factible",
                        "valor_objetivo": None,
                        "variables": None
                    }
                }
    
    # Extraer la solución final
    # Identificar variables básicas (columnas con exactamente un 1 y el resto ceros)
    variables_basicas = []
    valores_basicos = []
    
    for j in range(1, num_cols - 1):
        col_num = tableau_numerico[:, j]
        col_M = tableau_M[:, j]
        
        es_basica = False
        fila_uno = -1
        
        for i in range(num_filas):
            if abs(col_num[i] - 1) < 1e-10 and abs(col_M[i]) < 1e-10:
                # Verificar si el resto de la columna es cero
                es_cero_resto = True
                for k in range(num_filas):
                    if k != i and (abs(col_num[k]) > 1e-10 or abs(col_M[k]) > 1e-10):
                        es_cero_resto = False
                        break
                
                if es_cero_resto:
                    es_basica = True
                    fila_uno = i
                    break
        
        if es_basica:
            variables_basicas.append(j)
            valores_basicos.append(tableau_numerico[fila_uno, -1])
        else:
            variables_basicas.append(j)
            valores_basicos.append(0)
    
    # Preparar el resultado final
    resultado = {
        "status_text": "Óptimo" if iteracion <= max_iteraciones else "Número máximo de iteraciones alcanzado",
        "valor_objetivo": tableau_numerico[0, -1] if not es_minimizacion else -tableau_numerico[0, -1],
        "variables": []
    }
    
    # Extraer los valores de las variables originales
    for j in range(num_vars):
        idx = j + 1  # Índice en el tableau
        valor = 0
        
        if idx in variables_basicas:
            pos = variables_basicas.index(idx)
            valor = valores_basicos[pos]
        
        resultado["variables"].append({
            "nombre": f"x{j+1}",
            "valor": valor
        })
    
    return {
        "pasos": pasos,
        "metodo": "gran_m",
        "resultado_final": resultado
    }
