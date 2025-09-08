from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature columns exactly as in training (including spaces)
columns = ['texture_mean', 'perimeter_mean', 'concavity_mean', 
           'concave points_mean', 'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 'concavity_worst', 'concave points_worst']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    form_data = {}

    if request.method == 'POST':
        # Collect data from form
        form_data = {col: request.form.get(col.replace(' ', '_'), 0) for col in columns}

        # Convert to DataFrame
        data = pd.DataFrame({k: [float(v)] for k, v in form_data.items()})

        # Scale data
        data_scaled = scaler.transform(data)

        # Predict class and probability
        pred_class = model.predict(data_scaled)[0]
        pred_proba = model.predict_proba(data_scaled)[0]
        pred_class_prob_percent = pred_proba[pred_class] * 100

        prediction = 'Malignant' if pred_class == 1 else 'Benign'
        probability = f"{pred_class_prob_percent:.2f}%"

    return render_template('index.html', 
                           prediction=prediction, 
                           probability=probability,
                           form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
