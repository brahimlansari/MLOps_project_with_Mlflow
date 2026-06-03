import os
import joblib
import urllib.request
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'mlruns',
    '0',
    'e1afa9749f1745b8870327fe56b1538d',
    'artifacts',
    'model',
    'model.joblib',
)
PORT = int(os.getenv('PORT', 8000))
MLFLOW_URL = os.getenv('MLFLOW_UI_URL', 'http://127.0.0.1:5000')

FEATURES = [
    ('fixed_acidity', 'Fixed Acidity', '7.4'),
    ('volatile_acidity', 'Volatile Acidity', '0.70'),
    ('citric_acid', 'Citric Acid', '0.00'),
    ('residual_sugar', 'Residual Sugar', '1.9'),
    ('chlorides', 'Chlorides', '0.076'),
    ('free_sulfur_dioxide', 'Free Sulfur Dioxide', '11'),
    ('total_sulfur_dioxide', 'Total Sulfur Dioxide', '34'),
    ('density', 'Density', '0.9978'),
    ('pH', 'pH', '3.51'),
    ('sulphates', 'Sulphates', '0.56'),
    ('alcohol', 'Alcohol', '9.4'),
]

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None


def parse_input(data):
    values = {}
    features = []
    for key, _, default in FEATURES:
        raw = data.get(key)
        if raw is None or str(raw).strip() == '':
            raw = default
        values[key] = str(raw).strip()
        features.append(float(values[key]))
    return values, features


def predict_quality(features):
    if model is None:
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Lancez d'abord le pipeline ou placez le fichier model.joblib."
        )
    prediction = model.predict([features])
    return float(prediction[0])


def check_mlflow_status():
    try:
        with urllib.request.urlopen(MLFLOW_URL, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


@app.route('/', methods=['GET', 'POST'])
def index_page():
    prediction = None
    error = None
    values = {key: default for key, _, default in FEATURES}
    mlflow_online = check_mlflow_status()

    if request.method == 'POST':
        try:
            values, features = parse_input(request.form.to_dict(flat=True))
            prediction = predict_quality(features)
        except ValueError:
            error = 'Veuillez saisir des valeurs numériques valides pour tous les champs.'
        except Exception as exc:
            error = f'Erreur lors de la prédiction: {exc}'

    return render_template(
        'index.html',
        features=FEATURES,
        values=values,
        prediction=prediction,
        error=error,
        port=PORT,
        mlflow_url=MLFLOW_URL,
        mlflow_online=mlflow_online,
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict(flat=True)

        values, features = parse_input(data)
        prediction = predict_quality(features)
        return jsonify(status='success', prediction=prediction, inputs=values)
    except ValueError:
        return jsonify(status='error', message='Veuillez saisir des valeurs numériques valides pour tous les champs.'), 400
    except Exception as exc:
        return jsonify(status='error', message=str(exc)), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify(status='ok', model_loaded=model is not None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
