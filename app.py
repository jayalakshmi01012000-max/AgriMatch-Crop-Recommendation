import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
model = joblib.load("model.joblib")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def predict_crops(values):
    X = pd.DataFrame([values], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    top = sorted(zip(model.classes_, proba), key=lambda t: t[1], reverse=True)[:3]
    return [{"crop": c, "probability": round(float(p), 3)} for c, p in top]


@app.route("/", methods=["GET", "POST"])
def home():
    results, error, form = None, None, {}
    if request.method == "POST":
        form = request.form
        try:
            values = [float(form[f]) for f in FEATURES]
            results = predict_crops(values)
        except (KeyError, ValueError):
            error = "Please fill every field with a valid number."
    return render_template("index.html", features=FEATURES, results=results,
                           error=error, form=form)


@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    try:
        values = [float(data[f]) for f in FEATURES]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": f"Send numeric values for: {FEATURES}"}), 400
    return jsonify({"recommendations": predict_crops(values)})


if __name__ == "__main__":
    app.run(debug=True)
