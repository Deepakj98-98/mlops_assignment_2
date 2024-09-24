import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from ydata_profiling import ProfileReport
from dataprep.eda import create_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from autofeat import FeatureSelector
from autosklearn.classification import AutoSklearnClassifier
import matplotlib.pyplot as plt
import shap
import pickle

X, y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True)
X['survived']=y


report=ProfileReport(X, title="AutoEDA Report", explorative=True)
report.to_file("autosklearn_ydata_profiling_report.html")

report_dataprep = create_report(X)
report_dataprep.save("autosklearn_dataprep_report.html")

## Dropping features with maximum null values and features with no direct contribution like name and ticket.


drop_columns=['boat','body','home.dest','cabin','name','ticket']
X.drop(drop_columns,inplace=True,axis=1)
# X['age']=X['age'].fillna(X['age'].mean())
X=X.dropna(subset=["fare","embarked"],how='any')


## Creating test and train data


X_train,X_test,y_train,y_test=train_test_split(X.drop('survived',axis=1),X['survived'],test_size=0.2,shuffle=True,stratify=X['survived'])

## Considering pclass,sibsp,parch as numerical values and without feature engineering

numerical_minmax_features = ['pclass', 'sibsp','parch']
numerical_std_features = ['age', 'fare']
categorical_features = ['sex', 'embarked']  

# Numerical Preprocessing Pipeline
numerical_minmax_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())])

numerical_std_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Categorical Preprocessing Pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])

# Combine Both Preprocessing Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num_minmax', numerical_minmax_transformer, numerical_minmax_features),
        ('num_std', numerical_std_transformer, numerical_std_features),
        ('cat', categorical_transformer, categorical_features)])

X_train_1 = X_train.copy()
X_test_1 = X_test.copy()

X_train_1 = preprocessor.fit_transform(X_train_1)
X_test_1=preprocessor.transform(X_test_1)


## Considering pclass,sibsp,parch as numerical values and with feature engineering

X_train_2 = X_train_1.copy()
X_test_2 = X_test_1.copy()

fsel=FeatureSelector(problem_type="classification",verbose=1,featsel_runs=20)
X_train_2=fsel.fit_transform(X_train_2,y_train)

X_test_2=fsel.transform(X_test_2)

## Considering pclass,sibsp,parch as categorical values and without feature engineering

numerical_std_features = ['age', 'fare']
categorical_features_2 = ['sex', 'embarked','pclass', 'sibsp','parch']  

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Categorical Preprocessing Pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])

# Combine Both Preprocessing Pipelines
preprocessor_2 = ColumnTransformer(
    transformers=[
        ('num_std', numerical_std_transformer, numerical_std_features),
        ('cat', categorical_transformer, categorical_features_2)])


X_train_3 = X_train.copy()
X_test_3 = X_test.copy()


X_train_3 = preprocessor_2.fit_transform(X_train_3)
X_test_3 = preprocessor_2.transform(X_test_3)

## Considering pclass,sibsp,parch as categorical values and with feature engineering

X_train_4=X_train_3.copy()
X_test_4=X_test_3.copy()

fsel_1=FeatureSelector(problem_type="classification",verbose=1,featsel_runs=80)

X_train_4=fsel_1.fit_transform(X_train_4,y_train)
X_test_4=fsel_1.transform(X_test_4)

## Running Auto SKLearn to train all 4 model variants

def train_auto_sklearn(X_train, y_train):
    # Train the model using Auto-sklearn
    automl = AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=12,
        n_jobs=8,
        memory_limit=5120
    )
    automl.fit(X_train, y_train)
    
    
    
    return automl


def evaluate_model(model, X_test, y_test,name):
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f'{name} Test accuracy: {accuracy:.3f}')
    
    return accuracy


model_1=train_auto_sklearn(X_train_1, y_train)
model_2=train_auto_sklearn(X_train_2, y_train)
model_3=train_auto_sklearn(X_train_3, y_train)
model_4=train_auto_sklearn(X_train_4, y_train)

model1_eval=evaluate_model(model_1,X_test_1,y_test,"Model_1")
model2_eval=evaluate_model(model_2,X_test_2,y_test,"Model_2")
model3_eval=evaluate_model(model_3,X_test_3,y_test,"Model_3")
model4_eval=evaluate_model(model_4,X_test_4,y_test,"Model_4")

## Creating SHAP values and plotting for Model without AutoFeat and Model with AUtoFeat (Only 1 feature selected for male passenger)

explainer = shap.KernelExplainer(model_1.predict_proba,X_train_1)
shap_values = explainer.shap_values(X_test_1)
shap.summary_plot(shap_values, X_test_1,feature_names=preprocessor.get_feature_names_out())
plt.savefig('shap_summary_plot_without_feature_selection.png')
explainer_4 = shap.KernelExplainer(model_4.predict_proba,X_train_4)
shap_values_4 = explainer_4.shap_values(X_test_4)
shap.summary_plot(shap_values_4, X_test_4,feature_names=preprocessor_2.get_feature_names_out()[[fsel_1.original_columns_.index(i) for i in fsel_1.good_cols_]])
plt.savefig('shap_summary_plot_with_feature_selection.png')

## Saving best model and preprocessing pipeline

Model_1_file = 'python_function/function/Best_model.pkl'
with open(Model_1_file, 'wb') as file:
    pickle.dump(model_1, file)

Model_1_preprocessing = 'Best_model_preprocessing.pkl'
with open(Model_1_preprocessing, 'wb') as file:
    pickle.dump(preprocessor, file)

