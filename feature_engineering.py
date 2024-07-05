import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats

# Load data
df = pd.read_csv('data.csv')

# Data preprocessing - Imputation
imputer = SimpleImputer(strategy='mean')
df['lead_time'] = imputer.fit_transform(df['lead_time'].values.reshape(-1, 1))

# Data preprocessing - duplicate values
df.drop_duplicates(inplace=True)

# Data preprocessing - feature deletion
df.drop(['sku'], axis=1, inplace=True)

# Data preprocessing - encoding
encoder = LabelEncoder()
categorical_cols = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Data preprocessing - balancing data
smote = SMOTE(random_state=42)
X = df.drop('went_on_backorder', axis=1)
y = df['went_on_backorder']
X, y = smote.fit_resample(X, y)
df = pd.concat([X, y], axis=1)

# Data preprocessing - outliers
numerical_cols = ['national_inv', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month', 'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'pieces_past_due', 'local_bo_qty']
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < (Q1 - 1.5 * IQR)) |(df[col] > (Q3 + 1.5 * IQR)))]

# Save the transformed data frame
df.to_csv('cleaned_data.csv', index=False)

# Print the top 5 rows of cleaned dataframe
print(df.head())