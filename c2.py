import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

# === CONFIGURATION ===
file_path = r'C:\Users\Admin\Downloads\final_data_with_combined_ata (1).xlsx'
output_path = 'predicted_ata_levels.xlsx'

# === LOAD DATA ===
df = pd.read_excel(file_path)

# === CHECK REQUIRED COLUMNS ===
required_columns = ['DayDt', 'AC_REG_NO', 'ATA_LEVEL', 'FlightHours', 'Landings']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# === CLEAN & FORMAT DATE ===
df['DayDt'] = pd.to_datetime(df['DayDt'])

# === USER INPUT ===
ac_reg_no = input("Enter Aircraft Registration Number (e.g., VT-ANB): ").strip()
period_input = input("Enter period to predict (e.g., 2025-03, 2025Q2, or 2025): ").strip()

# === FUNCTIONS TO HANDLE PERIODS ===
def infer_period_type(p):
    if re.match(r"^\d{4}-\d{2}$", p):
        return "month"
    elif re.match(r"^\d{4}Q[1-4]$", p):
        return "quarter"
    elif re.match(r"^\d{4}$", p):
        return "year"
    else:
        raise ValueError("Invalid period format. Use YYYY-MM, YYYYQ#, or YYYY.")

def generate_target_dates(p):
    period_type = infer_period_type(p)
    if period_type == "month":
        target = datetime.strptime(p, "%Y-%m")
        return [target]
    elif period_type == "quarter":
        year = int(p[:4])
        q = int(p[-1])
        return [datetime(year, m, 1) for m in range((q - 1) * 3 + 1, q * 3 + 1)]
    elif period_type == "year":
        year = int(p)
        return [datetime(year, m, 1) for m in range(1, 13)]

target_dates = generate_target_dates(period_input)
min_target_date = min(target_dates)

# === FILTER HISTORICAL DATA FOR TRAINING ===
train_df = df[(df['AC_REG_NO'] == ac_reg_no) & (df['DayDt'] < min_target_date)].copy()

if train_df.empty:
    raise ValueError("No historical data available for this aircraft before the target period.")

# === Prepare multi-label target ===
# Group by DayDt and AC_REG_NO, collect ATA_LEVEL codes per date as list
grouped = train_df.groupby(['DayDt', 'AC_REG_NO'])['ATA_LEVEL'].apply(lambda x: list(set(x))).reset_index()

# Merge FlightHours and Landings by taking mean per date
agg_feats = train_df.groupby(['DayDt', 'AC_REG_NO']).agg({
    'FlightHours': 'mean',
    'Landings': 'mean'
}).reset_index()

# Merge grouped ATA_LEVEL list with features
train_data = pd.merge(grouped, agg_feats, on=['DayDt', 'AC_REG_NO'])

# Feature engineering - convert date to numeric (ordinal)
train_data['DayOrdinal'] = train_data['DayDt'].apply(lambda x: x.toordinal())

# Features matrix
X_train = train_data[['DayOrdinal', 'FlightHours', 'Landings']]

# Binarize multi-label ATA_LEVEL
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_data['ATA_LEVEL'])

# === Train multi-label classifier ===
model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)

# === Prepare prediction dataframe ===
# Estimate FlightHours and Landings using average from training data
avg_fh = train_data['FlightHours'].mean()
avg_ld = train_data['Landings'].mean()

pred_rows = []
for date in target_dates:
    pred_rows.append({
        'DayDt': date,
        'DayOrdinal': date.toordinal(),
        'FlightHours': avg_fh,
        'Landings': avg_ld
    })

pred_df = pd.DataFrame(pred_rows)
X_pred = pred_df[['DayOrdinal', 'FlightHours', 'Landings']]

# === Predict ===
Y_pred_prob = model.predict_proba(X_pred)  # probabilities for each ATA_LEVEL code

# Apply threshold to get binary predictions
threshold = 0.3
Y_pred = (Y_pred_prob >= threshold)

# === Convert to ATA_LEVEL code list per row ===
pred_lists = []
for row in Y_pred:
    indices = np.where(row)[0]
    codes = list(mlb.classes_[indices])
    pred_lists.append(codes)

# Add predictions to DataFrame
pred_df['Predicted_ATA_LEVELS'] = pred_lists
pred_df['AC_REG_NO'] = ac_reg_no

# Output only useful columns
output_df = pred_df[['AC_REG_NO', 'DayDt', 'Predicted_ATA_LEVELS']]

# === Save to Excel ===
output_df.to_excel(output_path, index=False)
print(f"\n✅ Predictions saved to '{output_path}'")
