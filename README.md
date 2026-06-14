# 📈 Sales Prediction Using Python
### Horizon TechX Data Science Internship — Task 4

---

## 📌 Project Overview
This project predicts future sales based on advertising spend across TV, Radio,
and Newspaper channels using regression-based machine learning models. It analyzes
how changes in advertising strategy impact sales outcomes.

## 📁 Project Structure
```
HorizonTechX_SalesPrediction/
│
├── Sales_Prediction.ipynb     # Main Jupyter Notebook (run this!)
├── sales_prediction.py        # Python script version
├── Advertising.csv            # Dataset
├── Sales_Prediction_Dashboard.png  # Static Dashboard Image
└── README.md                  # Project Documentation
```

## 📊 Steps Covered
1. Data Loading & Exploration
2. Data Preprocessing & Feature Engineering (Total Ad Spend, Spend Ratios)
3. Exploratory Data Analysis (EDA) — 6 visualizations
4. Model Building — Linear Regression & Random Forest Regressor
5. Model Evaluation — MAE, RMSE, R² Score
6. Feature Importance Analysis

## 📊 Visualizations Included
**EDA Dashboard:**
- TV Spend vs Sales
- Radio Spend vs Sales
- Newspaper Spend vs Sales
- Feature Correlation with Sales
- Sales Distribution
- Total Ad Spend vs Sales

**Model Evaluation Dashboard:**
- Model Accuracy Comparison (R² Score)
- Error Comparison (MAE & RMSE)
- Actual vs Predicted Sales
- Feature Importance Ranking

## 🛠️ Libraries Used
- Python 3
- Pandas
- NumPy
- Scikit-learn
- Plotly

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `Advertising.csv`
3. Paste the code from `Sales_Prediction.ipynb`
4. Run — both dashboards appear!

### Option 2 — Jupyter Notebook
1. Put all files in same folder
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn plotly
   ```
3. Run `Sales_Prediction.ipynb`

## 📂 Dataset Source
Kaggle: https://www.kaggle.com/datasets/bumba5341/advertisingcsv

## 🔑 Key Results
- **Best Model:** Random Forest Regressor (R² Score: 98.13%)
- Linear Regression achieved R² Score: 89.94%
- **TV advertising** has the strongest correlation with sales (0.78)
- Radio advertising also has a significant positive impact on sales
- Newspaper advertising has the weakest impact on sales

## 👤 Author

**Vasu Singhal** — B.Tech CSE (Data Science) — Bennett University — Horizon TechX Data Science Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vasu-singhal-46659a310)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Vasu-singhal01)

## 📄 License
MIT License — free to use and modify.
