from flask import Flask, render_template, request, jsonify
from models.lineal import resolver_modelo_lineal
from models.grafico import generar_metodo_grafico
from models.simplex import resolver_simplex_paso_a_paso
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resolver', methods=['POST'])
def resolver():
    try:
        # Obtener datos del formulario
        data = request.form.to_dict()
        
        # Procesar datos
        num_variables = int(data.get('num_variables', 2))
        num_restricciones = int(data.get('num_restricciones', 2))
        tipo_operacion = data.get('tipo_operacion', 'maximizar')
        
        # Obtener coeficientes de la función objetivo
        coef_objetivo = []
        for i in range(1, num_variables + 1):
            # Asegurarse de que los coeficientes se convierten correctamente a float
            coef_str = data.get(f'obj_coef_{i}', '0')
            try:
                coef = float(coef_str)
            except ValueError:
                coef = 0.0
            coef_objetivo.append(coef)
        
        # Obtener coeficientes de las restricciones
        coef_restricciones = []
        operadores = []
        lados_derechos = []
        
        for i in range(1, num_restricciones + 1):
            fila_coefs = []
            for j in range(1, num_variables + 1):
                coef = float(data.get(f'rest_coef_{i}_{j}', 0))
                fila_coefs.append(coef)
            
            coef_restricciones.append(fila_coefs)
            operadores.append(data.get(f'operador_{i}', '<='))
            lados_derechos.append(float(data.get(f'lado_derecho_{i}', 0)))
        
        # Preparar datos para el modelo
        datos_modelo = {
            'num_variables': num_variables,
            'num_restricciones': num_restricciones,
            'coef_objetivo': coef_objetivo,
            'tipo_operacion': tipo_operacion,
            'coef_restricciones': coef_restricciones,
            'operadores': operadores,
            'lados_derechos': lados_derechos
        }
        
        # Resolver el modelo
        resultados = resolver_modelo_lineal(datos_modelo)
        
        # Si el problema tiene 2 variables, generar el método gráfico
        metodo_grafico = None
        if num_variables == 2:
            metodo_grafico = generar_metodo_grafico(datos_modelo)
        
        # Renderizar la página de resultados
        return render_template('results.html', 
                              resultados=resultados, 
                              datos=datos_modelo,
                              metodo_grafico=metodo_grafico,
                              tiene_grafico=(num_variables == 2))
    
    except Exception as e:
        return render_template('results.html', 
                              resultados={'error': str(e)}, 
                              datos={},
                              metodo_grafico=None,
                              tiene_grafico=False)

@app.route('/simplex', methods=['POST'])
def simplex():
    try:
        # Obtener datos del formulario
        data = request.form.to_dict()
        
        # Procesar datos
        num_variables = int(data.get('num_variables', 2))
        num_restricciones = int(data.get('num_restricciones', 2))
        tipo_operacion = data.get('tipo_operacion', 'maximizar')
        
        # Obtener coeficientes de la función objetivo
        coef_objetivo = []
        for i in range(1, num_variables + 1):
            # Asegurarse de que los coeficientes se convierten correctamente a float
            coef_str = data.get(f'obj_coef_{i}', '0')
            try:
                coef = float(coef_str)
            except ValueError:
                coef = 0.0
            coef_objetivo.append(coef)
        
        # Obtener coeficientes de las restricciones
        coef_restricciones = []
        operadores = []
        lados_derechos = []
        
        for i in range(1, num_restricciones + 1):
            fila_coefs = []
            for j in range(1, num_variables + 1):
                coef = float(data.get(f'rest_coef_{i}_{j}', 0))
                fila_coefs.append(coef)
            
            coef_restricciones.append(fila_coefs)
            operadores.append(data.get(f'operador_{i}', '<='))
            lados_derechos.append(float(data.get(f'lado_derecho_{i}', 0)))
        
        # Preparar datos para el modelo
        datos_modelo = {
            'num_variables': num_variables,
            'num_restricciones': num_restricciones,
            'coef_objetivo': coef_objetivo,
            'tipo_operacion': tipo_operacion,
            'coef_restricciones': coef_restricciones,
            'operadores': operadores,
            'lados_derechos': lados_derechos
        }
        
        # Resolver el modelo usando el método Simplex paso a paso
        resultados_simplex = resolver_simplex_paso_a_paso(datos_modelo)
        
        # Renderizar la página de resultados del Simplex
        return render_template('simplex_results.html', 
                              resultados=resultados_simplex, 
                              datos=datos_modelo)
    
    except Exception as e:
        return render_template('simplex_results.html', 
                              resultados={'error': str(e)}, 
                              datos={})

if __name__ == '__main__':
    app.run(debug=True)
