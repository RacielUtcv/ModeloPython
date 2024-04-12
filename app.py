from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_DatosReales.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    productos = data['productos']
    predicciones = []
    for producto in productos:
        largo = producto['Largo(CM)']
        ancho = producto['Ancho(CM)']
        medidas = producto['Medidas(PULGADAS)']
        peso_predicho = modelo.predict([[largo, ancho, medidas]])[0]
        predicciones.append({
            'NombreProducto': producto['NombreProducto'],
            'attributes': {
                'length': str(largo),
                'height': str(medidas), 
                'width': str(ancho),  
                'weight': str(max(0, peso_predicho)),
                'mass_unit': 'KG',
                'distance_unit': 'CM'
            }
        })
    return jsonify({'predictions': predicciones})

if __name__ == '__main__':
    app.run(debug=True)
