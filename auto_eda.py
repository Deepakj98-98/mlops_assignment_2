# Load Dataset (Example: Titanic Dataset from Kaggle)
import pandas as pd
from ydata_profiling import ProfileReport
data = pd.read_csv("C:\\Users\\Deepak J Bhat\\Downloads\\train.csv")

profile = ProfileReport(data, title="AutoEDA Report", explorative=True)

# Save the report to an HTML file
profile.to_file("auto_eda_report.html")

# Display the report within a Jupyter Notebook (if using Jupyter)
profile.to_notebook_iframe()