from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

label_encoders = {}

label_encoders['education'] = LabelEncoder()
label_encoders['education'].fit(['Bachelor', 'Master', 'Associate', 'PhD'])

label_encoders['industry'] = LabelEncoder()
label_encoders['industry'].fit([
    'Automotive', 'Media', 'Education', 'Consulting', 'Healthcare', 'Gaming',
    'Government', 'Telecommunications', 'Manufacturing', 'Energy', 'Technology',
    'Real Estate', 'Finance', 'Transportation', 'Retail'
])

label_encoders['job_title'] = LabelEncoder()
label_encoders['job_title'].fit([
    'AI Research Scientist', 'AI Software Engineer', 'AI Specialist', 'NLP Engineer',
    'AI Consultant', 'AI Architect', 'Principal Data Scientist', 'Data Analyst',
    'Autonomous Systems Engineer', 'AI Product Manager', 'Machine Learning Engineer',
    'Data Engineer', 'Research Scientist', 'ML Ops Engineer', 'Robotics Engineer',
    'Head of AI', 'Deep Learning Engineer', 'Data Scientist', 'Machine Learning Researcher',
    'Computer Vision Engineer'
])

label_encoders['experience_level'] = LabelEncoder()
label_encoders['experience_level'].fit(['SE', 'EN', 'MI', 'EX'])

label_encoders['company_location'] = LabelEncoder()
label_encoders['company_location'].fit([
    'China', 'Canada', 'Switzerland', 'India', 'France', 'Germany', 'United Kingdom',
    'Singapore', 'Austria', 'Sweden', 'South Korea', 'Norway', 'Netherlands',
    'United States', 'Israel', 'Australia', 'Ireland', 'Denmark', 'Finland', 'Japan'
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        education = request.form['education']
        industry = request.form['industry']
        job_title = request.form['job_title']
        experience_level = request.form['experience_level']
        years_of_experience = int(request.form['years_of_experience'])
        company_location = request.form['company_location']

        education_enc = label_encoders['education'].transform([education])[0]
        industry_enc = label_encoders['industry'].transform([industry])[0]
        job_title_enc = label_encoders['job_title'].transform([job_title])[0]
        experience_enc = label_encoders['experience_level'].transform([experience_level])[0]
        company_location_enc = label_encoders['company_location'].transform([company_location])[0]

        features = np.array([[education_enc, industry_enc, job_title_enc, experience_enc, years_of_experience, company_location_enc]])
        print(f"[DEBUG] Encoded Features: {features}")
        prediction = model.predict(features)[0]
        print(f"[DEBUG] Predicted Salary: ${prediction:,.2f}")
        
        return render_template('index.html', prediction_text=f"Predicted Salary: ${prediction:,.2f}")
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return render_template('index.html', prediction_text=f"Error: {e}")
    
    
if __name__ == '__main__':
    app.run(debug=True)