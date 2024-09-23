from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('tpot_best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
app = Flask(__name__)

# Define a route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Example input: {"age": 30, "sex": "male", "fare": 100, "embarked": "S"}
    user_input = request.json 
    
    # Step 2: Create a DataFrame with the same structure as the training data
    input_df = pd.DataFrame({
        'Pclass': [user_input['Pclass']],
        'Age': [user_input['Age']],
        'SibSp': [user_input['SibSp']],
        'Parch': [user_input['Parch']],
        'Fare': [user_input['Fare']],
        'Sex': [user_input['Sex']],
        'Embarked': [user_input['Embarked']]
    })
    
    # Step 3: Preprocess the user input using the pre-trained pipeline
    processed_input = preprocessor.transform(input_df)
    
    # Step 4: Pass the preprocessed data to the model for predictions
    prediction = model.predict(processed_input)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
