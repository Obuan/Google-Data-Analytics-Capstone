import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('sales_data.csv')

# Data cleaning
data.dropna(inplace=True)

# Descriptive statistics
print(data.describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.boxplot(x='Region', y='Sales', data=data)
plt.title('Sales Distribution by Region')
plt.show()

# Correlation analysis
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Inferential statistics
from scipy.stats import ttest_ind

north_sales = data[data['Region'] == 'North']['Sales']
south_sales = data[data['Region'] == 'South']['Sales']

t_stat, p_val = ttest_ind(north_sales, south_sales)
print(f'T-statistic: {t_stat}, P-value: {p_val}')

# Regression analysis
import statsmodels.api as sm

X = data[['Sales']]
y = data['Profit']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())
