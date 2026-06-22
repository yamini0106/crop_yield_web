from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
import joblib
import os
import requests
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'agriyield_ai_super_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agri.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'home'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    crop = db.Column(db.String(50))
    region = db.Column(db.String(50))
    prediction = db.Column(db.Float)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

MODEL_PATH = os.path.join('model', 'crop_yield_model.pkl')
COLUMNS_PATH = os.path.join('model', 'model_columns.pkl')
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    hashed_pwd = generate_password_hash(data['password'])
    new_user = User(username=data['username'], password=hashed_pwd)
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'status': 'success', 'msg': 'Registration successful! Please login.'})
    except:
        return jsonify({'status': 'error', 'msg': 'Username already exists!'}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password, data['password']):
        login_user(user)
        return jsonify({'status': 'success', 'msg': 'Logged in successfully!'})
    return jsonify({'status': 'error', 'msg': 'Invalid username or password'}), 400

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/check_auth')
def check_auth():
    if current_user.is_authenticated:
        return jsonify({'logged_in': True, 'username': current_user.username})
    return jsonify({'logged_in': False})

@app.route('/get_weather', methods=['POST'])
def get_weather():
    city = request.json.get('city')
    if not city:
        return jsonify({'status': 'error', 'msg': 'City name required'}), 400
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_res = requests.get(geo_url).json()
        if not geo_res.get('results'):
            return jsonify({'status': 'error', 'msg': 'City not found'}), 404
        lat = geo_res['results'][0]['latitude']
        lon = geo_res['results'][0]['longitude']
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
        weather_res = requests.get(weather_url).json()
        current = weather_res.get('current', {})
        return jsonify({
            'status': 'success',
            'temp': current.get('temperature_2m', 25),
            'humidity': current.get('relative_humidity_2m', 60),
            'rainfall': current.get('precipitation', 0) * 100
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': 'Failed to fetch weather data'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_dict = {col: 0 for col in model_columns}
        
        numerical_features = ['Soil_pH', 'Rainfall_mm', 'Temperature_C', 'Humidity_pct', 'Fertilizer_Used_kg', 'Pesticides_Used_kg', 'Planting_Density']
        for feature in numerical_features:
            if feature in input_dict:
                input_dict[feature] = float(data[feature])
                
        categorical_mappings = {'Region': 'Region', 'Soil_Type': 'Soil_Type', 'Irrigation': 'Irrigation', 'Previous_Crop': 'Previous_Crop'}
        for frontend_key, backend_prefix in categorical_mappings.items():
            column_name = f"{backend_prefix}_{data[frontend_key]}"
            if column_name in input_dict:
                input_dict[column_name] = 1
                
        final_df = pd.DataFrame([input_dict])
        main_prediction = model.predict(final_df)[0]

        available_crops = ['Maize', 'Wheat', 'Rice', 'Barley']
        crop_yields = {}
        for crop in available_crops:
            temp_dict = input_dict.copy()
            for col in model_columns:
                if col.startswith('Crop_'):
                    temp_dict[col] = 0
            if f'Crop_{crop}' in temp_dict:
                temp_dict[f'Crop_{crop}'] = 1
            crop_pred = model.predict(pd.DataFrame([temp_dict]))[0]
            crop_yields[crop] = round(crop_pred, 2)
        best_crop = max(crop_yields, key=crop_yields.get)

        graph_data = {'fertilizer': [], 'yield': []}
        input_fert = float(data['Fertilizer_Used_kg'])
        for f in [0, input_fert*0.5, input_fert, input_fert*1.5, input_fert*2.0]:
            temp_dict = input_dict.copy()
            temp_dict['Fertilizer_Used_kg'] = f
            graph_data['fertilizer'].append(round(f, 1))
            graph_data['yield'].append(round(model.predict(pd.DataFrame([temp_dict]))[0], 2))

        importances = model.feature_importances_
        top_features = sorted(dict(zip(model_columns, importances)).items(), key=lambda x: x[1], reverse=True)[:5]
        explainability = [{"feature": feat.split('_')[-1], "importance": round(imp * 100, 2)} for feat, imp in top_features]

        if current_user.is_authenticated:
            new_pred = PredictionHistory(user_id=current_user.id, crop=data['Crop'], region=data['Region'], prediction=round(main_prediction, 2))
            db.session.add(new_pred)
            db.session.commit()

        return jsonify({
            'status': 'success',
            'prediction': round(main_prediction, 2),
            'selected_crop': data['Crop'],
            'crop_recommendations': crop_yields,
            'best_crop': best_crop,
            'graph_data': graph_data,
            'explainability': explainability
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/get_history')
def get_history():
    if current_user.is_authenticated:
        history = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.id.desc()).limit(5).all()
        return jsonify([{'crop': h.crop, 'region': h.region, 'yield': h.prediction} for h in history])
    return jsonify([])

@app.route('/delete_history', methods=['POST'])
@login_required
def delete_history():
    try:
        PredictionHistory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({'status': 'success', 'msg': 'History cleared successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': 'Failed to delete history'}), 500

if __name__ == '__main__':
    app.run(debug=True)