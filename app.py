from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open('model/agrishield_model.pkl', 'rb'))
le = pickle.load(open('model/label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        soil_ph = float(request.form['soil_ph'])
        humidity = float(request.form['humidity'])
        finance_score = float(request.form['finance_score'])

        # Create dataframe
        input_data = pd.DataFrame([[rainfall, temperature, soil_ph, humidity, finance_score]],
                                  columns=['Rainfall', 'Temperature', 'Soil_pH', 'Humidity', 'Finance_Score'])

        # Predict
        pred = model.predict(input_data)[0]
        label = le.inverse_transform([pred])[0]

        return render_template('result.html', prediction_text=f"Recommended Action: {label}")

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
