# Trafffic-congestion-prediction-using-ML

This repository contains the code and resources for an AI-based system designed to predict and visualize traffic congestion levels in Bangalore, India. The project leverages machine learning to provide real-time congestion insights, aiding urban planning and traffic management.

✨ Features
Traffic Congestion Prediction: Predicts real-time congestion levels based on various input parameters using a trained Machine Learning model.

Interactive Data Exploration: A Streamlit web application allows users to interactively explore relationships between different traffic parameters through dynamic plots.

Model Evaluation: Includes code for evaluating model performance (RMSE, R2 score) and analyzing potential overfitting/underfitting.

Data Preprocessing: Handles data cleaning, feature engineering, and categorical encoding.

🛠️ Technologies Used
Python

Pandas: Data manipulation and analysis.

NumPy: Numerical operations.

Scikit-learn: Machine learning model (RandomForestRegressor), data splitting, and evaluation metrics.

Matplotlib: Static plotting.

Seaborn: Enhanced statistical data visualization.

Streamlit: For building the interactive web application.

Joblib: For saving and loading the trained machine learning model.

Pyngrok (Optional): For creating public URLs for the Streamlit app when running in environments like Google Colab.

📊 Dataset
The project utilizes the Banglore_traffic_Dataset.csv dataset. This dataset includes various features influencing traffic, such as:

Traffic Volume

Average Speed

Travel Time Index

Road Capacity Utilization

Incident Reports

Public Transport Usage

Traffic Signal Compliance

Weather Conditions

Target Variable: Congestion Level (a continuous numeric value).

🧠 Machine Learning Model
The core predictive component of this project is a RandomForestRegressor.

Model Type: Ensemble Learning (Regression)

Why RandomForestRegressor?

Robustness to outliers and noisy data.

Ability to handle non-linear relationships.

Good generalization capabilities and reduced risk of overfitting compared to single decision trees.

Training: The model is trained on 80% of the dataset, with 20% reserved for testing (test_size=0.2, random_state=42).

Evaluation Metric: Root Mean Squared Error (RMSE) is used to quantify prediction accuracy.
