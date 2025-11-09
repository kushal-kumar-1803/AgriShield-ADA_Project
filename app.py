from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ------------------------------
# Load model and encoder
# ------------------------------
MODEL_PATH = os.path.join("model", "agrishield_model.pkl")
ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
label_encoder = pickle.load(open(ENCODER_PATH, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        rainfall = request.form.get("rainfall")
        temperature = request.form.get("temperature")
        soil_ph = request.form.get("soil_ph")
        humidity = request.form.get("humidity")
        financial_score = request.form.get("financial_score")

        # Validate inputs
        if not all([rainfall, temperature, soil_ph, humidity, financial_score]):
            return render_template(
                "result.html",
                prediction_text="⚠️ Please fill in all fields before submitting."
            )

        # Prepare input for model
        data = pd.DataFrame([{
            "Rainfall": float(rainfall),
            "Temperature": float(temperature),
            "Soil_pH": float(soil_ph),
            "Humidity": float(humidity),
            "Financial_Score": float(financial_score)
        }])

        # Ensure correct feature order
        data = data.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make prediction
        pred = model.predict(data)[0]
        result = label_encoder.inverse_transform([pred])[0]

        return render_template(
            "result.html",
            prediction_text=f"🌾 Recommended Action: {result}"
        )

    except Exception as e:
        return render_template("result.html", prediction_text=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
