import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Delete 'lead_time' column
df = df.drop('lead_time', axis=1)

# Handle NaN values
df = df.dropna()

# Delete 'sku' column
df = df.drop('sku', axis=1)

# Label encoding
le = LabelEncoder()
categorical_cols = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Check for missing values and drop them
df = df.dropna()

# Balance 'went_on_backorder' using SMOTE
smote = SMOTE(random_state=42)
df['went_on_backorder'] = df['went_on_backorder'].astype(int)
df_smote, y_smote = smote.fit_resample(df.drop('went_on_backorder', axis=1), df['went_on_backorder'])
df = pd.concat([df_smote, y_smote], axis=1)

# Outliers using Z-Score
numerical_cols = ['national_inv', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month', 'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'pieces_past_due', 'local_bo_qty']
for col in numerical_cols:
    df = df[np.abs(stats.zscore(df[col])) < 3]

# Print the result
print(df.head())

# Save the transformed data frame
df.to_csv('cleaned_data.csv', index=False)
