import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from functools import wraps

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'loan_model.joblib')

# Load trained model pipeline if available
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Configurable API Key for RESTful API endpoint security
API_KEY = os.getenv("API_KEY", "prod-secret-key-12345")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        client_key = request.headers.get("X-API-KEY")
        if client_key != API_KEY:
            return jsonify({"error": "Unauthorized access. Invalid or missing X-API-KEY."}), 401
        return f(*args, **kwargs)
    return decorated

def process_prediction(loan, mortdue, value, reason, job, yoj, derog, delinq, clage, ninq, clno, debtinc):
    ltv = loan / (value + 1e-5)
    net_equity = value - mortdue

    input_df = pd.DataFrame([{
        'LOAN': loan, 'MORTDUE': mortdue, 'VALUE': value,
        'REASON': reason, 'JOB': job, 'YOJ': yoj,
        'DEROG': derog, 'DELINQ': delinq, 'CLAGE': clage,
        'NINQ': ninq, 'CLNO': clno, 'DEBTINC': debtinc,
        'LOAN_TO_VALUE_RATIO': ltv, 'NET_EQUITY': net_equity
    }])

    if model:
        prediction = int(model.predict(input_df)[0])
        default_prob = float(model.predict_proba(input_df)[0][1]) * 100
    else:
        # Fallback heuristic calculation if model binary is missing
        risk_factor = (derog * 15) + (delinq * 20) + (debtinc * 1.2) + (ltv * 30) - (yoj * 1.5)
        default_prob = max(1.0, min(99.0, risk_factor))
        prediction = 1 if default_prob >= 45.0 else 0

    credit_score = int(850 - ((default_prob / 100.0) * 550))
    score_percentage = round(((credit_score - 300) / 550) * 100, 1)

    return {
        'prediction': prediction,
        'verdict': 'CREDIT SCHEME APPROVED' if prediction == 0 else 'HIGH RISK - REJECTED',
        'default_prob': round(default_prob, 1),
        'credit_score': credit_score,
        'score_percentage': score_percentage,
        'score_status': 'Excellent' if credit_score >= 740 else ('Good' if credit_score >= 670 else ('Fair' if credit_score >= 580 else 'Poor')),
        'ltv': round(ltv * 100, 2),
        'net_equity': round(net_equity, 2)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    form_data = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            res = process_prediction(
                loan=float(request.form.get('LOAN', 25000)),
                mortdue=float(request.form.get('MORTDUE', 75000)),
                value=float(request.form.get('VALUE', 120000)),
                reason=request.form.get('REASON', 'HomeImp'),
                job=request.form.get('JOB', 'Other'),
                yoj=float(request.form.get('YOJ', 5)),
                derog=float(request.form.get('DEROG', 0)),
                delinq=float(request.form.get('DELINQ', 0)),
                clage=float(request.form.get('CLAGE', 180)),
                ninq=float(request.form.get('NINQ', 1)),
                clno=float(request.form.get('CLNO', 15)),
                debtinc=float(request.form.get('DEBTINC', 32.5))
            )
            result = res
        except Exception as e:
            result = {'error': str(e)}

    return render_template('index.html', result=result, form_data=form_data)

@app.route('/api/predict', methods=['POST'])
@require_api_key
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON payload required'}), 400

    try:
        res = process_prediction(
            loan=float(data.get('LOAN', 25000)),
            mortdue=float(data.get('MORTDUE', 75000)),
            value=float(data.get('VALUE', 120000)),
            reason=data.get('REASON', 'HomeImp'),
            job=data.get('JOB', 'Other'),
            yoj=float(data.get('YOJ', 5)),
            derog=float(data.get('DEROG', 0)),
            delinq=float(data.get('DELINQ', 0)),
            clage=float(data.get('CLAGE', 180)),
            ninq=float(data.get('NINQ', 1)),
            clno=float(data.get('CLNO', 15)),
            debtinc=float(data.get('DEBTINC', 32.5))
        )
        return jsonify({
            'status': 'success',
            'data': res
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)