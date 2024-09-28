from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
tpot_model = joblib.load('tpot_best_model.pkl')
tpot_preprocessor = joblib.load('preprocessor.pkl')
autosklearn_model = joblib.load('python_function/function/Best_model.pkl')
autosklearn_preprocessor = joblib.load('python_function/function/Best_model_preprocessing.pkl')
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

    #TPOT prediction pipeline
    
    # Step 3: Preprocess the user input using the pre-trained pipeline
    tpot_processed_input = tpot_preprocessor.transform(input_df)
    
    # Step 4: Pass the preprocessed data to the model for predictions
    tpot_prediction = tpot_model.predict(tpot_processed_input)


    # Auto-Sklearn prediction pipeline

    ASK_processed_input = autosklearn_preprocessor.transform(input_df[['pclass','sex','age','sibsp','parch','fare','embarked']])
    
    ASK_prediction = autosklearn_model.predict(ASK_processed_input)

    return jsonify({'TPOT_prediction': int(tpot_prediction[0]),
                    'Auto-Sklearn prediction':int(ASK_prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
