# ============================================================
#  Sales Prediction Using Python
#  Horizon TechX Data Science Internship — Task 4
#  Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv
# ============================================================

# ── Step 1: Install & Import Libraries ──
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'plotly', 'scikit-learn', '-q'])

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")


# ── Step 2: Load & Explore Dataset ──
df = pd.read_csv('Advertising.csv')

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print("="*55)
print("        ADVERTISING SALES DATASET — OVERVIEW")
print("="*55)
print(f"  Total Records    : {df.shape[0]}")
print(f"  Total Features   : {df.shape[1]}")
print(f"  Missing Values   : {df.isnull().sum().sum()}")
print(f"  Duplicate Rows   : {df.duplicated().sum()}")
print("="*55)
print()
print("📋 First 5 Records:")
print(df.head())
print()
print("📈 Statistical Summary:")
print(df.describe())


# ── Step 3: Data Preprocessing & Feature Engineering ──

# Check correlations
correlation = df.corr()

print("✅ Correlation with Sales:")
print(correlation['Sales'].sort_values(ascending=False))
print()

# Total Ad Spend feature
df['Total_Ad_Spend'] = df['TV'] + df['Radio'] + df['Newspaper']

# Spend ratio features
df['TV_Ratio']  = df['TV']  / df['Total_Ad_Spend']
df['Radio_Ratio'] = df['Radio'] / df['Total_Ad_Spend']

print("📌 New Features Added:")
print("   Total_Ad_Spend — Sum of TV, Radio, Newspaper spend")
print("   TV_Ratio       — Proportion of budget spent on TV")
print("   Radio_Ratio    — Proportion of budget spent on Radio")
print()
print("✅ Feature Engineering Done!")


# ── Step 4: EDA Dashboard ──

BG   = '#0d1117'
CARD = '#161b22'
BLUE = '#58a6ff'
RED  = '#f78166'
GREEN= '#3fb950'
TEXT = '#e6edf3'
ORG  = '#ffa657'
PURP = '#d2a8ff'

fig1 = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        '📺 TV Spend vs Sales',
        '📻 Radio Spend vs Sales',
        '📰 Newspaper Spend vs Sales',
        '🔥 Feature Correlation with Sales',
        '📊 Sales Distribution',
        '💰 Total Ad Spend vs Sales'
    ),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "bar"}],
        [{"type": "histogram"}, {"type": "scatter"}]
    ],
    vertical_spacing=0.12, horizontal_spacing=0.10
)

fig1.add_trace(go.Scatter(x=df['TV'], y=df['Sales'], mode='markers',
    marker=dict(color=BLUE, size=7, opacity=0.7)), row=1, col=1)

fig1.add_trace(go.Scatter(x=df['Radio'], y=df['Sales'], mode='markers',
    marker=dict(color=GREEN, size=7, opacity=0.7)), row=1, col=2)

fig1.add_trace(go.Scatter(x=df['Newspaper'], y=df['Sales'], mode='markers',
    marker=dict(color=ORG, size=7, opacity=0.7)), row=2, col=1)

corr_sales = correlation['Sales'].drop('Sales').sort_values(ascending=False)
fig1.add_trace(go.Bar(x=corr_sales.index, y=corr_sales.values,
    marker_color=[GREEN, ORG, RED],
    text=[f'{v:.2f}' for v in corr_sales.values], textposition='outside'), row=2, col=2)

fig1.add_trace(go.Histogram(x=df['Sales'], nbinsx=20, marker_color=PURP, opacity=0.8), row=3, col=1)

fig1.add_trace(go.Scatter(x=df['Total_Ad_Spend'], y=df['Sales'], mode='markers',
    marker=dict(color=RED, size=7, opacity=0.7)), row=3, col=2)

fig1.update_layout(
    title=dict(
        text='📈 Sales Prediction — Exploratory Data Analysis<br><sup>Horizon TechX Data Science Internship | Task 4</sup>',
        font=dict(size=18, color=TEXT), x=0.5
    ),
    height=1200,
    plot_bgcolor=CARD, paper_bgcolor=BG,
    font=dict(color=TEXT),
    showlegend=False
)
fig1.update_xaxes(gridcolor='#21262d')
fig1.update_yaxes(gridcolor='#21262d')
fig1.show()
print("✅ EDA Dashboard Complete!")


# ── Step 5: Build ML Models ──

features = ['TV', 'Radio', 'Newspaper']
X = df[features]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1 — Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Model 2 — Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Evaluation
lr_r2   = r2_score(y_test, lr_pred)
rf_r2   = r2_score(y_test, rf_pred)
lr_mae  = mean_absolute_error(y_test, lr_pred)
rf_mae  = mean_absolute_error(y_test, rf_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("="*50)
print("      MODEL EVALUATION RESULTS")
print("="*50)
print(f"  Linear Regression")
print(f"    R²   : {lr_r2*100:.2f}%")
print(f"    MAE  : {lr_mae:.4f}")
print(f"    RMSE : {lr_rmse:.4f}")
print()
print(f"  Random Forest Regressor")
print(f"    R²   : {rf_r2*100:.2f}%")
print(f"    MAE  : {rf_mae:.4f}")
print(f"    RMSE : {rf_rmse:.4f}")
print("="*50)

best_model = "Random Forest Regressor" if rf_r2 > lr_r2 else "Linear Regression"
print(f"\n🏆 Best Model: {best_model}")

# Linear Regression Coefficients
print("\n📊 Linear Regression Coefficients:")
for feat, coef in zip(features, lr.coef_):
    print(f"   {feat:10s} : {coef:.4f}")
print(f"   Intercept  : {lr.intercept_:.4f}")


# ── Step 6: Model Evaluation Dashboard ──

feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '🏆 Model Accuracy (R² Score)',
        '📉 Error Comparison (MAE & RMSE)',
        '🎯 Actual vs Predicted Sales',
        '🔍 Feature Importance (Random Forest)'
    ),
    specs=[
        [{"type": "bar"},     {"type": "bar"}],
        [{"type": "scatter"}, {"type": "bar"}]
    ],
    vertical_spacing=0.15, horizontal_spacing=0.12
)

fig2.add_trace(go.Bar(
    x=['Linear Regression', 'Random Forest'],
    y=[lr_r2*100, rf_r2*100],
    marker_color=[BLUE, GREEN],
    text=[f'{lr_r2*100:.1f}%', f'{rf_r2*100:.1f}%'],
    textposition='outside'
), row=1, col=1)

fig2.add_trace(go.Bar(
    x=['LR MAE', 'RF MAE', 'LR RMSE', 'RF RMSE'],
    y=[lr_mae, rf_mae, lr_rmse, rf_rmse],
    marker_color=[BLUE, GREEN, RED, ORG],
    text=[f'{v:.3f}' for v in [lr_mae, rf_mae, lr_rmse, rf_rmse]],
    textposition='outside'
), row=1, col=2)

best_pred = rf_pred if rf_r2 > lr_r2 else lr_pred
fig2.add_trace(go.Scatter(x=list(y_test), y=list(best_pred), mode='markers',
    marker=dict(color=GREEN, size=8, opacity=0.7)), row=2, col=1)
fig2.add_trace(go.Scatter(
    x=[float(y_test.min()), float(y_test.max())],
    y=[float(y_test.min()), float(y_test.max())],
    mode='lines', line=dict(color=RED, dash='dash', width=2)
), row=2, col=1)

fig2.add_trace(go.Bar(
    x=feat_imp.values[::-1], y=feat_imp.index[::-1],
    orientation='h', marker_color=PURP,
    text=[f'{v:.3f}' for v in feat_imp.values[::-1]],
    textposition='outside'
), row=2, col=2)

fig2.update_layout(
    title=dict(
        text=f'🏆 Model Evaluation Dashboard — Sales Prediction<br><sup>Horizon TechX Data Science Internship | Task 4 | Best Model: {best_model}</sup>',
        font=dict(size=16, color=TEXT), x=0.5
    ),
    height=900,
    plot_bgcolor=CARD, paper_bgcolor=BG,
    font=dict(color=TEXT),
    showlegend=False
)
fig2.update_xaxes(gridcolor='#21262d')
fig2.update_yaxes(gridcolor='#21262d')
fig2.show()

print("\n✅ Sales Prediction — All Steps Complete!")
print(f"   Best Model     : {best_model}")
print(f"   R² Score       : {max(lr_r2, rf_r2)*100:.2f}%")
