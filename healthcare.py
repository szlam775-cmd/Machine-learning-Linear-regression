import pandas as pd
from scipy.stats import chi2_contingency

# Set display options
pd.set_option("display.max_columns", 50)
pd.set_option("display.precision", 2)
pd.set_option('display.width', None)

# ============================================
# STEP 1: READ THE DATA (ONLY ONCE!)
# ============================================
df = pd.read_excel(r"C:\Users\Raiyan\Downloads\diabetic_data_QMH_Club_Fest_2025.xlsx", sheet_name="Sheet1")

print("Original data:")
print(df.head())
print(f"Shape: {df.shape}")
df.info()

# ============================================
# STEP 2: RENAME COLUMNS
# ============================================
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
    'V48': 'med_change_status',
    'V49': 'diabetic_med_given',
    'V50': 'readmission_status'
}

df.rename(columns=column_mapping, inplace=True)

print("\n✓ Columns renamed successfully!")
print(f"Dataset shape: {df.shape}")
print("\nNew column names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# ============================================
# STEP 3: HANDLE MISSING VALUES
# ============================================
df['glucose_test_result'] = df['glucose_test_result'].fillna('Not Tested')
df['A1C_result'] = df['A1C_result'].fillna('Not Tested')

print(f"\nComplete records: {len(df)}")
print(f"No missing values: {df.isnull().sum().sum() == 0}")

# ============================================
# STEP 4: FIX DATA TYPES
# ============================================

# Numerical columns
numerical_cols = [
    'encounter_code',
    'patient_code',
    'adm_type_code',
    'discharge_type',
    'adm_src_code',
    'hospital_days',
    'lab_test_count',
    'procedure_count',
    'medication_count',
    'outpatient_visits',
    'emergency_visits',
    'inpatient_visits',
    'diagnosis_total'
]

for col in numerical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\n✓ Converted {len(numerical_cols)} numerical columns")

# Categorical columns
nominal_cols = [
    'ethnic_group',
    'sex_identity',
    'age_group',
    'body_weight',
    'insurance_code',
    'provider_specialty',
    'diagnosis_primary',
    'diagnosis_secondary',
    'diagnosis_tertiary',
    'glucose_test_result',
    'A1C_result',
    'medication_1', 'medication_2', 'medication_3', 'medication_4',
    'medication_5', 'medication_6', 'medication_7', 'medication_8',
    'medication_9', 'medication_10', 'medication_11', 'medication_12',
    'medication_13', 'medication_14', 'medication_15', 'medication_16',
    'medication_17', 'medication_18', 'medication_19', 'medication_20',
    'medication_21', 'medication_22', 'medication_23',
    'med_change_status',
    'diabetic_med_given',
    'readmission_status'
]

for col in nominal_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f"✓ Converted {len(nominal_cols)} nominal (categorical) columns")

# ============================================
# STEP 5: ADD DISCHARGE & ADMISSION DESCRIPTIONS
# ============================================

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

# Map the codes to descriptions (NOW using correct column names!)
df['discharge_type_desc'] = df['discharge_type'].map(discharge_type_dict)
df['admission_source_desc'] = df['adm_src_code'].map(admission_source_dict)


# ============================================
# ADD ADMISSION TYPE DESCRIPTIONS (V7)
# ============================================

admission_type_dict = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Available"
}

# Map the codes to descriptions
df['admission_type_desc'] = df['adm_type_code'].map(admission_type_dict)

print("\n✓ Added discharge, admission source, and admission type descriptions")

# ============================================
# STEP 6: VERIFY DATA
# ============================================

print("\n" + "="*50)
print("DATA TYPE SUMMARY")
print("="*50)

print(f"\nNumerical columns ({len(df.select_dtypes(include=['int64', 'float64']).columns)}):")
print(df.select_dtypes(include=['int64', 'float64']).columns.tolist())

print(f"\nNominal columns ({len(df.select_dtypes(include='category').columns)}):")
print(df.select_dtypes(include='category').columns.tolist())

print("\n" + "="*50)
print("SAMPLE DATA WITH NEW DESCRIPTIONS")
print("="*50)
print(df[['discharge_type', 'discharge_type_desc', 'adm_src_code', 'admission_source_desc']].head(10))

print("\n" + "="*50)
print("DATA QUALITY CHECKS")
print("="*50)

print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print("\nNumerical columns summary:")
print(df.describe())

print("\nKey nominal column distributions:")
print("\nReadmission status:")
print(df['readmission_status'].value_counts())

print("\nAge groups:")
print(df['age_group'].value_counts().sort_index())

print("\nSex identity:")
print(df['sex_identity'].value_counts())

print("\n" + "="*50)
print("SAMPLE DATA WITH ALL DESCRIPTIONS")
print("="*50)
print(df[['adm_type_code', 'admission_type_desc', 
          'discharge_type', 'discharge_type_desc', 
          'adm_src_code', 'admission_source_desc']].head(10))

# ============================================
# STEP 7: ANALYSIS
# ============================================


print("\n" + "="*50)
print("ANALYSIS RESULTS")
print("="*50)

# Basic statistics
print("\nAverage hospital days:", df['hospital_days'].mean())
print("Correlation between meds and emergency visits:", 
      df['medication_count'].corr(df['emergency_visits']))

# Readmission by age
print("\nReadmission by age group:")
print(pd.crosstab(df['age_group'], df['readmission_status'], normalize='index'))

# Readmission by sex
print("\nReadmission by sex:")
print(pd.crosstab(df['sex_identity'], df['readmission_status'], normalize='index'))

# Medications by age
print("\nAverage medications by age group:")
print(df.groupby('age_group')['medication_count'].mean().sort_index())

# High medication analysis
df['high_medication'] = (df['medication_count'] > df['medication_count'].median()).astype(int)

print("\nReadmission by age + medication count:")
print(pd.crosstab([df['age_group'], df['high_medication']], 
                   df['readmission_status'], 
                   normalize='index'))

# Elderly with high medications
elderly_high_meds = df[(df['age_group'].isin(['[60-70)', '[70-80)', '[80-90)'])) & 
                       (df['medication_count'] > 15)]

print("\n<30 day readmission for elderly with 15+ medications:")
print(elderly_high_meds['readmission_status'].value_counts(normalize=True))

# Chi-square test
contingency_table = pd.crosstab(df['age_group'], df['readmission_status'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test: p-value = {p_value:.4f}")
if p_value < 0.05:
    print("✓ Age and readmission are statistically related!")

# ============================================
# STEP 8: DISCHARGE & ADMISSION ANALYSIS
# ============================================

print("\n" + "="*50)
print("DISCHARGE TYPE & ADMISSION SOURCE ANALYSIS")
print("="*50)

print("\n=== DISCHARGE TYPE ANALYSIS ===")
print("\nMost common discharge types:")
print(df['discharge_type_desc'].value_counts().head(10))

print("\n=== ADMISSION SOURCE ANALYSIS ===")
print("\nMost common admission sources:")
print(df['admission_source_desc'].value_counts().head(10))

print("\n Analysis complete!")

# NEW: Admission Type Analysis
print("\n=== ADMISSION TYPE ANALYSIS ===")
print("\nAdmission type distribution:")
print(df['admission_type_desc'].value_counts())

print("\nReadmission rate by admission type:")
for adm_type in df['admission_type_desc'].value_counts().index:
    subset = df[df['admission_type_desc'] == adm_type]
    readmission_rate = ((subset['readmission_status'] != 'NO').sum() / len(subset) * 100)
    print(f"  {adm_type}: {readmission_rate:.1f}% readmitted")

# Crosstab analysis
print("\nDetailed readmission by admission type:")
print(pd.crosstab(df['admission_type_desc'], df['readmission_status'], normalize='index'))



# ============================================
# MEDICATION COUNT BY READMISSION STATUS
# ============================================

print("\n" + "="*50)
print("MEDICATION COUNT: FIVE-NUMBER SUMMARY BY READMISSION STATUS")
print("="*50)

# Group by readmission status and calculate five-number summary
readmission_groups = df.groupby('readmission_status')['medication_count']

print("\nDetailed statistics for each readmission group:")
print("-" * 50)

for group_name, group_data in readmission_groups:
    print(f"\n{group_name}:")
    print(f"  Count:    {len(group_data)}")
    print(f"  Min:      {group_data.min()}")
    print(f"  Q1:       {group_data.quantile(0.25)}")
    print(f"  Median:   {group_data.median()}")
    print(f"  Q3:       {group_data.quantile(0.75)}")
    print(f"  Max:      {group_data.max()}")
    print(f"  Mean:     {group_data.mean():.2f}")
    print(f"  Std Dev:  {group_data.std():.2f}")

# Create a summary table
summary_table = pd.DataFrame({
    'Min': readmission_groups.min(),
    'Q1': readmission_groups.quantile(0.25),
    'Median': readmission_groups.median(),
    'Q3': readmission_groups.quantile(0.75),
    'Max': readmission_groups.max(),
    'Mean': readmission_groups.mean(),
    'Count': readmission_groups.count()
})

print("\n" + "="*50)
print("SUMMARY TABLE")
print("="*50)
print(summary_table)

# Calculate IQR and identify outliers
print("\n" + "="*50)
print("OUTLIER ANALYSIS")
print("="*50)

for group_name, group_data in readmission_groups:
    q1 = group_data.quantile(0.25)
    q3 = group_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = group_data[(group_data < lower_bound) | (group_data > upper_bound)]
    
    print(f"\n{group_name}:")
    print(f"  IQR: {iqr}")
    print(f"  Lower bound: {lower_bound}")
    print(f"  Upper bound: {upper_bound}")
    print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(group_data)*100:.1f}%)")
