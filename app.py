from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define all possible states for one-hot encoding
states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Extract data from form
        try:
            name = request.form['name']
            gender = request.form['gender']
            age = int(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            health = request.form['health']
            checkup = request.form['checkup']
            diabetes = request.form['diabetes']
            smoking = request.form['smoking']
            exercise = request.form['exercise']
            mental_health = request.form['mental_health']
            physical_health = request.form['physical_health']
            aids = request.form['aids']
            arthritis = request.form['arthritis']
            state = request.form['state']
            metropolitan_status = request.form['metropolitan_status']
            urban_status = request.form['urban_status']
            alcohol = request.form['alcohol']

            # Calculate BMI
            bmi = weight / (height ** 2)

            # Map categorical variables to numerical values
            gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
            health_map = {'Excellent': 0, 'Very good': 1, 'Good': 2, 'Fair': 3, 'Poor': 4}
            checkup_map = {'Within past year': 0, 'Within past 2 years': 1, 'Within past 5 years': 2, '5 or more years ago': 3}
            diabetes_map = {'No': 0, 'Yes (during pregnancy)': 1, 'Yes (not during pregnancy)': 2}
            smoking_map = {'Current smoker': 0, 'Former smoker': 1, 'Never smoked': 2}
            exercise_map = {'Active': 0, 'Not active': 1}
            health_status_map = {'Good': 0, 'Fair': 1, 'Poor': 2}
            hiv_aids_map = {'No': 0, 'Yes': 1}
            arthritis_map = {'No': 0, 'Yes': 1}
            metropolitan_status_map = {'Metropolitan': 0, 'Non-metropolitan': 1}
            urban_status_map = {'Urban': 0, 'Rural': 1}
            alcohol_map = {'No': 0, 'Yes': 1}
            state_map = {state: idx for idx, state in enumerate(states)}

            # Create the feature array in the correct order
            data = np.array([[
                state_map[state],
                metropolitan_status_map[metropolitan_status],
                urban_status_map[urban_status],
                weight,
                height,
                age,
                checkup_map[checkup],
                health_map[health],
                exercise_map[exercise],
                gender_map[gender],
                arthritis_map[arthritis],
                diabetes_map[diabetes],
                health_status_map[mental_health],
                smoking_map[smoking],
                alcohol_map[alcohol],
                health_status_map[physical_health],
                hiv_aids_map[aids],
                bmi
            ]])

            # Debugging: Print the shape and values of the data
            print(f"Data shape: {data.shape}")
            print(f"Data values: {data}")

            # Ensure the input data has the correct shape
            if data.shape[1] == model.coef_.shape[1]:
                # Predict the risk
                risk_prediction = model.predict(data)[0]
                return render_template('result.html', name=name, risk_prediction=risk_prediction)
            else:
                return f"Error: Incorrect number of features provided. Expected {model.coef_.shape[1]}, got {data.shape[1]}.", 400
        except Exception as e:
            print(f"Error: {e}")
            return "Error processing request.", 400

if __name__ == '__main__':
    app.run(debug=True)
