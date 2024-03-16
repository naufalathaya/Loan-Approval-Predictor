from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load('model.pkl')

# Load the fitted StandardScaler
scaler = joblib.load('scaler.pkl')

# Render the home page with the loan application form
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Extract input features from the form
        gender = (request.form['gender'])
        married = (request.form['married'])
        dependents = (request.form['dependents'])
        education = (request.form['education'])
        self_employed = (request.form['self_employed'])
        term = (request.form['term'])
        credit_history = (request.form['credit_history'])
        area = (request.form['area'])
        applicant_income = (request.form['applicant_income'])
        coapplicant_income = (request.form['coapplicant_income'])
        loan_amount = (request.form['loan_amount'])

        # Standardize numerical features
        X = np.array([[gender, married, term, dependents, self_employed, education, area, credit_history, applicant_income, coapplicant_income, loan_amount]])
        X_scaled = scaler.transform(X)

        # Make prediction
        prediction = model.predict(X_scaled)[0]

        # Mapping prediction to human-readable format
        prediction = 'Approved' if prediction == 1 else 'Not Approved'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
