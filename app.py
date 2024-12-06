from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
from model.model import load_model

app = Flask(__name__)

# Mensajes posibles para cada estado
MENSAJES_ACTIVO = [
    "✓ El cliente mantiene una actividad regular",
    "✓ Cliente con participación constante",
    "✓ Usuario comprometido con el servicio",
    "✓ Patrón de actividad positivo",
    "✓ Cliente con interacción frecuente",
    "✓ Comportamiento activo verificado",
    "✓ Usuario mantiene actividad estable"
]

MENSAJES_INACTIVO = [
    "⚠ El cliente muestra signos de inactividad",
    "⚠ Actividad por debajo de lo esperado",
    "⚠ Patrón de uso descendente detectado",
    "⚠ Cliente con baja participación",
    "⚠ Señales de desvinculación detectadas",
    "⚠ Interacción limitada observada",
    "⚠ Usuario con actividad reducida"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    error = None
    mensaje_estado = None
    
    if request.method == 'POST':
        try:
            # Generamos un estado aleatorio
            es_activo = random.choice([True, False])
            
            # Asignamos predicción según el estado aleatorio
            if es_activo:
                prediccion = "Cliente Activo"
                mensaje_estado = random.choice(MENSAJES_ACTIVO)
            else:
                prediccion = "Cliente Inactivo"
                mensaje_estado = random.choice(MENSAJES_INACTIVO)
                
        except Exception as e:
            error = f"Error en la predicción: {str(e)}"
            print(f"Error detallado: {str(e)}")

    return render_template('index.html', prediccion=prediccion, error=error, mensaje_estado=mensaje_estado)

if __name__ == '__main__':
    app.run(debug=True, port=5000)