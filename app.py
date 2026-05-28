from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Show current directory
print("Current directory:", os.getcwd())

# Full path to model
model_path = os.path.abspath("model.pkl")
print("Loading model from:", model_path)

# Load model ONCE
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Feature order MUST match training data
        features = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        # Convert to numpy array
        final_input = np.array([features])

        # Prediction
        prediction = model.predict(final_input)

        return render_template(
            "index.html",
            prediction_text="Heart Disease" if prediction[0] == 1 else "No Heart Disease"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)