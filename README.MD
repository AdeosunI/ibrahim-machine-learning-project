# Traffic Accident Prediction Using Time-Series Models (ARMA and LSTM)

## Overview

This project uses historical traffic accident data from the US Accidents Dataset (Kaggle) to forecast daily accident counts. Two types of models are implemented:

- ARMA (classical statistical method)
- LSTM (deep learning method)

The project compares the models based on their prediction accuracy (measured by RMSE) across multiple simulation trials.

## How to Install

1. Ensure you have **Python 3.8 or higher** installed.
2. Clone the project repository or download the project folder.
3. Install the required Python packages by running:

   ```
   pip install -r requirements.txt
   ```

Required libraries include:

- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- tensorflow

## How to Run the Software

1. Download the dataset:

   - Visit [US Accidents Dataset on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
   - Sign in or create a Kaggle account if needed.
   - Download the `US_Accidents_March23.csv` file.
   - Save it into the project directory where the code is located.

2. Place the `US_Accidents_March23.csv` dataset in the project directory.
3. Open a terminal in the project folder.
4. Run:

   ```
   python3 traffic_accident_prediction.py
   ```

This will:

- Preprocess and clean the dataset
- Train and evaluate ARMA and LSTM models
- Perform 5 experimental trials for each model
- Display RMSE results
- Generate visual plots comparing model performance

## How the Experiments Work

- **Training Data:** 2022 and 2023 accident records
- **Testing Data:** 2020 and 2021 accident records
- Each model is trained and evaluated over 5 trials.
- RMSE is calculated for each model and compared.

## Outputs

- RMSE scores printed to the console for each trial
- Bar chart comparing average RMSE between ARMA and LSTM
- Line plot of actual vs. predicted accident counts

## Files Included

- `traffic_accident_prediction.py` — Main script
- `requirements.txt` — Python dependencies
- `README.md` — This file
