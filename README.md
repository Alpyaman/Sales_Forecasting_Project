
# 📈 Sales Forecasting with LightGBM – Time Series Regression Project

![Sales Forecasting Thumbnail](A_grid_of_four_data_visualizations_related_to_sale.png)

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)

This project forecasts weekly sales for retail stores using historical sales data, store information, and external features. It includes advanced feature engineering, model tuning, and result evaluation.

---

## 📁 Project Structure
```
sales-forecasting/
├── data/                           # Input CSV files
├── preprocess.py                   # Merges & transforms raw data
├── train_model.py                  # Baseline model (Random Forest)
├── train_lgbm.py                   # LightGBM model + hyperparameter tuning
├── predict.py                      # Generates predictions and saves output
├── A_grid_of_four_data_visualizations_related_to_sale.png  # Project thumbnail
└── README.md                       # Project overview
```

---

## 🔍 Feature Engineering
- Merged sales, stores, and features data
- Created lag features (`Lag_1`, `Lag_4`)
- Added rolling statistics (`Rolling_Mean_4`, `Rolling_Std_4`)
- Encoded store types
- Extracted date parts: `Year`, `Month`, `Week`, `DayOfWeek`, `IsWeekend`

---

## 🤖 Models Used
- **Random Forest Regressor** (baseline)
- **LightGBM Regressor** (tuned using `RandomizedSearchCV`)

**Metrics:**  
- RMSE (Root Mean Squared Error)  
- R² Score

---

## 📈 Results
- Achieved strong predictive performance with tuned LightGBM model
- Identified most important predictors via feature importance plots
- Plotted predicted vs actual sales

---

## 📦 Requirements
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run
```bash
python preprocess.py
python train_lgbm.py
python predict.py
```

---

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## 👤 Author
Created by [Alp Yaman](https://github.com/yourgithubusername)
