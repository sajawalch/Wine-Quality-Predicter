import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load files relative to this script, so it works no matter where you run it from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pickle(filename):
    with open(os.path.join(BASE_DIR, filename), "rb") as f:
        return pickle.load(f)


model = load_pickle("wine_quality_model.pkl")
scaler = load_pickle("wine_quality_scaler.pkl")

# (feature name used in training, HTML form field name)
# Order MUST match the training columns.
FEATURES = [
    ("fixed acidity", "fixed_acidity"),
    ("volatile acidity", "volatile_acidity"),
    ("citric acid", "citric_acid"),
    ("residual sugar", "residual_sugar"),
    ("chlorides", "chlorides"),
    ("free sulfur dioxide", "free_sulfur_dioxide"),
    ("total sulfur dioxide", "total_sulfur_dioxide"),
    ("density", "density"),
    ("pH", "pH"),
    ("sulphates", "sulphates"),
    ("alcohol", "alcohol"),
]


@app.route("/")
def home():
    return render_template("index.html", prediction=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {col: float(request.form[field]) for col, field in FEATURES}
    except KeyError as e:
        return render_template("index.html", prediction=None,
                               error=f"Missing form field: {e.args[0]}")
    except ValueError:
        return render_template("index.html", prediction=None,
                               error="Please enter valid numbers in every field.")

    # Same column names and order as training
    input_data = pd.DataFrame([values], columns=[col for col, _ in FEATURES])

    scaled_data = scaler.transform(input_data)
    raw = float(model.predict(scaled_data)[0])

    # Quality is a whole-number score (3-8), so also show a rounded value
    return render_template(
        "index.html",
        prediction=round(raw, 2),
        rounded=int(round(raw)),
        error=None,
    )


if __name__ == "__main__":
    app.run(debug=True)  # set debug=False when deploying