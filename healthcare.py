import pandas as pd
pd.set_option("display.max_columns", 50)
pd.set_option("display.precision", 2)

df = pd.read_excel(r"C:\Users\Raiyan\Downloads\diabetic_data_QMH_Club_Fest_2025.xlsx", sheet_name="Sheet1")

print(df.head())
print(df.shape)
df.info()

import pandas as pd

df = pd.read_excel(r"C:\Users\Raiyan\Downloads\diabetic_data_QMH_Club_Fest_2025.xlsx")

column_mapping = {
    'V1': 'encounter_code',
    'V2': 'patient_code',
    'V3': 'ethnic_group',
    'V4': 'sex_identity',
    'V5': 'age_group',
    'V6': 'body_weight',
    'V7': 'adm_type_code',
    'V8': 'discharge_type',
    'V9': 'adm_src_code',
    'V10': 'hospital_days',
    'V11': 'insurance_code',
    'V12': 'provider_specialty',
    'V13': 'lab_test_count',
    'V14': 'procedure_count',
    'V15': 'medication_count',
    'V16': 'outpatient_visits',
    'V17': 'emergency_visits',
    'V18': 'inpatient_visits',
    'V19': 'diagnosis_primary',
    'V20': 'diagnosis_secondary',
    'V21': 'diagnosis_tertiary',
    'V22': 'diagnosis_total',
    'V23': 'glucose_test_result',
    'V24': 'A1C_result',
    # Medication columns (V25-V47)
    'V25': 'medication_1',
    'V26': 'medication_2',
    'V27': 'medication_3',
    'V28': 'medication_4',
    'V29': 'medication_5',
    'V30': 'medication_6',
    'V31': 'medication_7',
    'V32': 'medication_8',
    'V33': 'medication_9',
    'V34': 'medication_10',
    'V35': 'medication_11',
    'V36': 'medication_12',
    'V37': 'medication_13',
    'V38': 'medication_14',
    'V39': 'medication_15',
    'V40': 'medication_16',
    'V41': 'medication_17',
    'V42': 'medication_18',
    'V43': 'medication_19',
    'V44': 'medication_20',
    'V45': 'medication_21',
    'V46': 'medication_22',
    'V47': 'medication_23',
    # Final columns
    'V48': 'med_change_status',
    'V49': 'diabetic_med_given',
    'V50': 'readmission_status'
}
df.rename(columns=column_mapping, inplace=True)

print("✓ Columns renamed successfully!")
print(f"\nDataset shape: {df.shape}")
print("\nNew column names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(df.head())

# Complete handling code
df['glucose_test_result'] = df['glucose_test_result'].fillna('Not Tested')
df['A1C_result'] = df['A1C_result'].fillna('Not Tested')

# Now you have no missing data and can use all 101,766 records!
print(f"Complete records: {len(df)}")
print(f"No missing values: {df.isnull().sum().sum() == 0}")

# ============================================
# NUMERICAL COLUMNS (Quantitative - can do math on them)
# ============================================
numerical_cols = [
    'encounter_code',
    'patient_code',
    'adm_type_code',
    'discharge_type',
    'adm_src_code',
    'hospital_days',           # Number of days in hospital
    'lab_test_count',          # Count of lab tests
    'procedure_count',         # Count of procedures
    'medication_count',        # Count of medications
    'outpatient_visits',       # Number of visits
    'emergency_visits',        # Number of visits
    'inpatient_visits',        # Number of visits
    'diagnosis_total'          # Total diagnoses
]

# Convert to numeric
for col in numerical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"✓ Converted {len(numerical_cols)} numerical columns")

# ============================================
# NOMINAL COLUMNS (Categorical - labels/categories)
# ============================================
nominal_cols = [
    # Demographics
    'ethnic_group',            # Asian, Caucasian, African American, etc.
    'sex_identity',            # Male, Female
    'age_group',               # [0-10), [10-20), etc.
    'body_weight',             # Weight categories
    
    # Administrative
    'insurance_code',          # Insurance type
    'provider_specialty',      # Doctor specialty
    
    # Medical diagnoses
    'diagnosis_primary',       # Primary diagnosis code
    'diagnosis_secondary',     # Secondary diagnosis code
    'diagnosis_tertiary',      # Tertiary diagnosis code
    
    # Test results
    'glucose_test_result',     # Normal, High, Not Tested, etc.
    'A1C_result',              # Normal, High, Not Tested, etc.
    
    # Medications (23 medication columns)
    'medication_1', 'medication_2', 'medication_3', 'medication_4',
    'medication_5', 'medication_6', 'medication_7', 'medication_8',
    'medication_9', 'medication_10', 'medication_11', 'medication_12',
    'medication_13', 'medication_14', 'medication_15', 'medication_16',
    'medication_17', 'medication_18', 'medication_19', 'medication_20',
    'medication_21', 'medication_22', 'medication_23',
    
    # Outcomes
    'med_change_status',       # Changed, Not Changed
    'diabetic_med_given',      # Yes, No
    'readmission_status'       # Yes, No, >30 days
]

# Convert to category (nominal)
for col in nominal_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f"✓ Converted {len(nominal_cols)} nominal (categorical) columns")

# ============================================
# VERIFY THE CONVERSIONS
# ============================================
print("\n" + "="*50)
print("DATA TYPE SUMMARY")
print("="*50)

print(f"\nNumerical columns ({len(df.select_dtypes(include=['int64', 'float64']).columns)}):")
print(df.select_dtypes(include=['int64', 'float64']).columns.tolist())

print(f"\nNominal columns ({len(df.select_dtypes(include='category').columns)}):")
print(df.select_dtypes(include='category').columns.tolist())

print("\n" + "="*50)
print("SAMPLE DATA")
print("="*50)
print(df.head())

print("\n" + "="*50)
print("DETAILED INFO")
print("="*50)
df.info()

# ============================================
# USEFUL CHECKS
# ============================================
print("\n" + "="*50)
print("DATA QUALITY CHECKS")
print("="*50)

# Check for any remaining missing values
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Summary statistics for numerical columns
print("\nNumerical columns summary:")
print(df.describe())

# Check unique values in key nominal columns
print("\nKey nominal column distributions:")
print(f"\nReadmission status:")
print(df['readmission_status'].value_counts())

print(f"\nAge groups:")
print(df['age_group'].value_counts().sort_index())

print(f"\nSex identity:")
print(df['sex_identity'].value_counts())

# Numerical analysis
print("Average hospital days:", df['hospital_days'].mean())
print("Correlation between meds and visits:", 
      df['medication_count'].corr(df['emergency_visits']))

# Nominal analysis
print("\nReadmission by age group:")
print(pd.crosstab(df['age_group'], df['readmission_status'], normalize='index'))

print("\nReadmission by sex:")
print(pd.crosstab(df['sex_identity'], df['readmission_status'], normalize='index'))

# For your hypothesis: "Older patients with more medications"
print("\nAverage medications by age group:")
print(df.groupby('age_group')['medication_count'].mean().sort_index())

# 1. Combine age and medication count to test your hypothesis directly
df['high_medication'] = (df['medication_count'] > df['medication_count'].median()).astype(int)

print("Readmission by age + medication count:")
print(pd.crosstab([df['age_group'], df['high_medication']], 
                   df['readmission_status'], 
                   normalize='index'))

# 2. Look at specific <30 day readmissions for older patients with many meds
elderly_high_meds = df[(df['age_group'].isin(['[60-70)', '[70-80)', '[80-90)'])) & 
                       (df['medication_count'] > 15)]

print("\n<30 day readmission for elderly with 15+ medications:")
print(elderly_high_meds['readmission_status'].value_counts(normalize=True))

# 3. Statistical test
from scipy.stats import chi2_contingency

# Test if age and readmission are related
contingency_table = pd.crosstab(df['age_group'], df['readmission_status'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test: p-value = {p_value:.4f}")
if p_value < 0.05:
    print(" Age and readmission are statistically related!")

import pandas as pd

# Read your data
df = pd.read_excel(r"C:\Users\Raiyan\Downloads\diabetic_data_QMH_Club_Fest_2025.xlsx")

# Define the dictionaries
discharge_type_dict = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    16: "Discharged/transferred/referred to another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home. Medicaid only, hospice.",
    20: "Expired in a medical facility. Medicaid only, hospice.",
    21: "Expired, place unknown. Medicaid only, hospice.",
    22: "Discharged/transferred to another rehab fac including rehab units of a hospital.",
    23: "Discharged/transferred to a long term care hospital",
    24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
    28: "Discharged/transferred to a federal health care facility.",
    29: "Discharged/transferred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
    30: "Discharged/transferred to a Critical Access Hospital (CAH)."
}

admission_source_dict = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a hospital",
    5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Available",
    17: "NULL",
    18: "Transfer from Another Home Health Agency",
    19: "Readmission to Same Home Health Agency",
    20: "Not Mapped",
    21: "Unknown/Invalid",
    22: "Transfer from hospital inpt/same fac result in a sep claim",
    23: "Born inside this hospital",
    24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center",
    26: "Transfer from Hospice"
}

# Map the codes to descriptions
df['discharge_type_desc'] = df['V8'].map(discharge_type_dict)
df['admission_source_desc'] = df['V9'].map(admission_source_dict)

# View the results
print("Sample data with descriptions:")
print(df[['V8', 'discharge_type_desc', 'V9', 'admission_source_desc']].head(10))


print("=== DISCHARGE TYPE ANALYSIS ===")
print("\nMost common discharge types:")
print(df['discharge_type_desc'].value_counts().head(10))

print("\n=== ADMISSION SOURCE ANALYSIS ===")
print("\nMost common admission sources:")
print(df['admission_source_desc'].value_counts().head(10))
