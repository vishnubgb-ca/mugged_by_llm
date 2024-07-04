import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats

# Load data
df = pd.read_csv('data.csv')

# Delete 'lead_time' column
df = df.drop(['lead_time'], axis=1)

# Handle NaN values
df = df.dropna()

# Delete duplicate values
df = df.drop_duplicates()

# Delete 'sku' column
df = df.drop(['sku'], axis=1)

# Label encoding
le = LabelEncoder()
categorical_cols = ['potential_issue', 'deck_risk', 'ppap_risk', 'oe_constraint', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Balancing data
smote = SMOTE(random_state=42)
X = df.drop('went_on_backorder', axis=1)
y = df['went_on_backorder']
X, y = smote.fit_resample(X, y)
df = pd.concat([X, y], axis=1)

# Outliers using IQR method
numerical_cols = ['national_inv', 'forecast_3_month', 'in_transit_qty', 'forecast_6_month', 'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'pieces_past_due', 'local_bo_qty']
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

# Save cleaned data
df.to_csv('cleaned_data.csv', index=False)

# Print top 5 rows of cleaned dataframe
print(df.head())