
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

users = {}

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    if 'username' in session:
        return redirect('/predict-form')
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect('/predict-form')
        return "Invalid credentials! <a href='/login'>Try again</a>"
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return "Username already exists! <a href='/signup'>Try again</a>"
        users[username] = password
        return redirect('/login')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

@app.route('/predict-form')
def predict_form():
    if 'username' not in session:
        return redirect('/login')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect('/login')

    input_dict = {
        'SeniorCitizen': 1 if request.form['SeniorCitizen'] == 'Yes' else 0,
        'Partner': 1 if request.form['Partner'] == 'Yes' else 0,
        'tenure': int(request.form['Tenure']),
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'gender_Male': 1 if request.form['Gender'] == 'Male' else 0,
        'Contract_One year': 1 if request.form['Contract'] == 'One year' else 0,
        'Contract_Two year': 1 if request.form['Contract'] == 'Two year' else 0,
        'InternetService_DSL': 1 if request.form['InternetService'] == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if request.form['InternetService'] == 'Fiber optic' else 0,
        'PaperlessBilling_Yes': 1 if request.form['PaperlessBilling'] == 'Yes' else 0
    }

    feature_order = [
        'SeniorCitizen', 'Partner', 'tenure', 'MonthlyCharges',
        'gender_Male', 'Contract_One year', 'Contract_Two year',
        'InternetService_DSL', 'InternetService_Fiber optic',
        'PaperlessBilling_Yes'
    ]

    input_values = [input_dict[feature] for feature in feature_order]
    scaled_input = scaler.transform([input_values])

    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0][1] * 100

    reasons = [
        {'label': 'Long Response Times', 'value': 42.06},
        {'label': 'Inexperienced Staff / Bad service', 'value': 31.97},
        {'label': 'High Charges / Interest', 'value': 19.5},
        {'label': 'Too Many Documents', 'value': 8.36}
    ]

    return render_template('result.html',
                           prediction=prediction,
                           confidence=confidence,
                           reasons=reasons)

if __name__ == '__main__':
    app.run(debug=True)
