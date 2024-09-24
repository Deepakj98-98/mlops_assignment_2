# Required Libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, classification_report
from lime import lime_tabular
import matplotlib.pyplot as plt

# Load Dataset (Example: Titanic Dataset from Kaggle)
data = pd.read_csv("./train.csv")

# Data Preprocessing
# Handle missing values
# Numerical and categorical columns
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_cols = ['Pclass', 'Sex', 'Embarked']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Missing value imputation using SimpleImputer
# Scaling numeric features using StandardScaler
# Encoding categorical variables using OneHotEncoder
X = data.drop('Survived', axis=1)  # Replace 'Target' with your actual target column
y = data['Survived']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit and transform the training data
X_train = preprocessor.fit_transform(X_train)
# Extract the feature names from the ColumnTransformer
categorical_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
    ['Pclass', 'Sex', 'Embarked'])

# Combine with numerical feature names
final_feature_names = numerical_cols + list(categorical_feature_names)
# Print the final feature names
print(final_feature_names)
joblib.dump(preprocessor, './python_function/function/preprocessor.pkl')
X_test = preprocessor.transform(X_test)
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_test: {X_train.shape}')

# Ensure that X_test is a DataFrame with proper feature names
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test, columns=final_feature_names)

# Initialize TPOT classifier
tpot = TPOTClassifier(
    generations=5,  # Number of generations to run
    population_size=20,  # Number of pipelines to test in each generation
    verbosity=2,  # Level of verbosity (2 shows more details)
    random_state=42
)

# Train TPOT
tpot.fit(X_train, y_train)
# tpot.export('tpot_best_model.py')
# Predict on the test set
y_pred = tpot.predict(X_test)

# Calculate accuracy and display classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'TPOT Best Model Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# To display the best pipeline found by TPOT
print("\nTPOT Best Pipeline:")
print(tpot.fitted_pipeline_)
joblib.dump(tpot.fitted_pipeline_, './python_function/function/tpot_best_model.pkl')
# Display all non-dominated pipelines tried by TPOT
print("\nAll Non-Dominated Pipelines Tried by TPOT:")
for pipeline in tpot.pareto_front_fitted_pipelines_.keys():
    print(f'Pipeline: {pipeline}')

# ==================================
# LIME EXPLANATIONS
# ==================================


print(X_train.shape[1])
# Initialize the LIME explainer
class_names = ['Survived']
explainer = lime_tabular.LimeTabularExplainer(
    X_train,  # Pass X_train as NumPy array
    feature_names=final_feature_names,  # Give custom feature names
    class_names=['Survived'],  # Modify based on your class names
    mode='classification'
)

# Get explanation for a specific instance
i = 0
instance = X_test.iloc[i].values
# Generate explanations for all classes
for i in range(len(class_names)):
    exp = explainer.explain_instance(instance, tpot.predict_proba, num_features=7, top_labels=len(class_names))
    fig = exp.as_pyplot_figure(label=i)
    plt.title(f'Local explanation for class {class_names[i]}')
    plt.show()
