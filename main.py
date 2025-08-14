import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_accounts = 100
num_months = 12
# Generate synthetic data for sales performance
data = {
    'AccountID': np.arange(1, num_accounts + 1),
    'Month': np.tile(np.arange(1, num_months + 1), num_accounts),
    'Sales': np.random.randint(500, 5000, size=num_accounts * num_months),
    'MarketTrend': np.random.normal(1, 0.1, size=num_accounts * num_months) # Simulate market fluctuations
}
df = pd.DataFrame(data)
# Introduce some at-risk accounts by decreasing their sales over time
at_risk_accounts = np.random.choice(df['AccountID'].unique(), size=int(0.2 * num_accounts), replace=False)
for account in at_risk_accounts:
    df.loc[(df['AccountID'] == account), 'Sales'] -= (df.loc[(df['AccountID'] == account), 'Month'] * 200)
# --- 2. Data Cleaning and Feature Engineering ---
# No explicit cleaning needed for this synthetic data.
# --- 3. Predictive Modeling ---
# Prepare data for modeling
X = df[['Month', 'MarketTrend']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model (simple example, can be replaced with more sophisticated models)
model = LinearRegression()
model.fit(X_train, y_train)
# Predict sales for the test set
predictions = model.predict(X_test)
# --- 4. Risk Identification ---
# Define a threshold for at-risk accounts (e.g., sales below a certain percentage of predicted sales)
risk_threshold = 0.8  # Accounts with sales below 80% of predicted sales are at risk
df['PredictedSales'] = model.predict(X)
df['AtRisk'] = (df['Sales'] < df['PredictedSales'] * risk_threshold).astype(int)
# --- 5. Visualization ---
# Plot sales trends for at-risk accounts
at_risk_df = df[df['AtRisk'] == 1]
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Sales', hue='AccountID', data=at_risk_df)
plt.title('Sales Trend for At-Risk Accounts')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.tight_layout()
plt.savefig('at_risk_accounts.png')
print("Plot saved to at_risk_accounts.png")
#Plot overall sales trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Sales', data=df)
plt.title('Overall Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.tight_layout()
plt.savefig('overall_sales_trend.png')
print("Plot saved to overall_sales_trend.png")
# --- 6. Output ---
# (In a real-world scenario, this section would involve creating a dashboard)
print("At-risk accounts identified. See plots for details.")