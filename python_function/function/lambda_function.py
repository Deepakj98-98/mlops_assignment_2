import json
import logging
import os

import joblib
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

env = os.environ["environment"]

model = joblib.load('tpot_best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')


def lambda_handler(event, context):
    try:
        print("Inside mlops-lambda Lambda !!!", event)
        print("Inside mlops-lambda Lambda !!!", event['body'])

        # Example input: {"age": 30, "sex": "male", "fare": 100, "embarked": "S"}
        user_input = json.load(event['body'])

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

        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': int(prediction[0])}),
            'headers': {
                "Access-Control-Allow-Origin": "*"
            }
        }
    except Exception as e:
        print('Error Occurred in lambda  Record :: ', e)
        logger.exception(e)
    finally:
        logger.info('End of mlops-lambda lambda')


