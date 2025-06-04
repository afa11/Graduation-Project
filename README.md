# Graduation-Project
Graduation Project - Boğaziçi University
Midterm Presentation 11.04.2025

# Graduation Project: Predictive Maintenance for MetroPT3
 
This repository contains the complete workflow for a predictive maintenance study using the MetroPT3 dataset. The goal of the project is to anticipate failures in metro systems by analyzing sensor data and applying machine learning models.
 
## 🔍 Project Overview
 
The workflow is structured as a series of Jupyter notebooks that reflect the progression from data exploration to model selection and evaluation. The project mainly focuses on classifying failure conditions using both logistic regression and random forest models, evaluated on their early warning capabilities and predictive performance.
 
## 📁 Repository Structure
 
- **1_metropt3_exploring_data_start.ipynb**  
  Initial inspection of the MetroPT3 dataset, identifying sensor structure, failure annotations, and general data quality.
 
- **2_exploring_data_and_data_preprocessing.ipynb**  
  Detailed preprocessing pipeline including time filtering, label creation, and feature extraction.
 
- **3a_metropt3_modelling.ipynb & 3b_metropt3_modelling.ipynb**  
  Early attempts at modeling failure conditions using baseline classifiers.
 
- **3c_metropt_logistic_regression.ipynb**  
  First focused logistic regression model for predicting failures.
 
- **4_model_for_midterm_presentation.ipynb**  
  Midterm milestone notebook summarizing findings and presenting early models.
 
- **5_prefinal.ipynb**  
  Pre-final integration notebook where major modeling decisions were consolidated.
 
- **6_model_selection_random_forest.ipynb**  
  Random Forest model building, hyperparameter tuning, and evaluation.
 
- **7_model_selection_logistic_regression.ipynb**  
  Logistic regression counterpart of notebook 6, with model comparisons and early warning insights.
 
- **8 - last_models.ipynb**  
  Final models and comparative analysis of performance across classifiers.
 
- **deneme_7_model_selection_logistic_regression.ipynb**  
  Experimental run for logistic regression, likely an intermediate testing version.
 
- **failure_time.txt**  
  Annotated failure timestamps used for labeling failure windows in the data.
 
- **mpt_functions.py**  
  Utility functions used throughout the notebooks for preprocessing, feature extraction, and evaluation.
 
- **Data Description_Metro-3.pdf**  
  Official dataset documentation outlining sensor types, recording structure, and units.
 
- **metropt2.ipynb**  
  Earlier or alternative version of data exploration, not central to the final workflow.
 
- **MetroPT-3 (1).zip**  
  Contains raw or preprocessed data in compressed form (not needed for execution if already extracted).
 
- **ARIMA/**, **Other Efforts/**, **Varmax/**  
  Contain exploratory time-series forecasting attempts that were not included in the final implementation.
 
## 📌 Notes
 
- This project emphasizes early detection of failures — notebooks 6, 7, and 8 are particularly crucial for the final results.
- The implementation focuses on classification-based approaches over time-series forecasting.
