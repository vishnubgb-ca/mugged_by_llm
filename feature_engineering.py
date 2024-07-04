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
categorical_cols = ['potential_issue', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'deck_risk', 'rev_stop', 'went_on_backorder']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Balancing data using SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(df.drop('went_on_backorder', axis=1), df['went_on_backorder'])
df = pd.concat([X_smote, y_smote], axis=1)

# Outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Save the transformed data frame
df.to_csv('cleaned_data.csv', index=False)

# Print the top 5 rows of cleaned dataframe
print(df.head())