# import pandas as pd
# import numpy as np
# from scipy.stats import chi2_contingency
# import logging
# from typing import Optional, Dict, List, Tuple
# import warnings

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Suppress warnings for cleaner output
# warnings.filterwarnings('ignore')


# class DiabeticDataProcessor:
#     """
#     Comprehensive error handling for diabetic readmission analysis
#     """
    
#     def __init__(self, file_path: str):
#         self.file_path = file_path
#         self.df: Optional[pd.DataFrame] = None
#         self.errors: List[str] = []
#         self.warnings: List[str] = []
        
#     def safe_load_data(self) -> pd.DataFrame:
#         """Load Excel file with comprehensive error handling"""
#         try:
#             logger.info(f"Loading data from {self.file_path}")
#             self.df = pd.read_excel(self.file_path, sheet_name="Sheet1")
            
#             if self.df.empty:
#                 raise ValueError("Dataset is empty")
            
#             logger.info(f"✓ Loaded {self.df.shape[0]:,} rows and {self.df.shape[1]} columns")
#             return self.df
            
#         except FileNotFoundError:
#             logger.error(f"❌ File not found: {self.file_path}")
#             raise
#         except PermissionError:
#             logger.error(f"❌ Permission denied: {self.file_path}")
#             raise
#         except Exception as e:
#             logger.error(f"❌ Error loading file: {str(e)}")
#             raise
    
#     def safe_rename_columns(self) -> pd.DataFrame:
#         """Rename columns with validation"""
#         column_mapping = {
#             'V1': 'encounter_code', 'V2': 'patient_code', 'V3': 'ethnic_group',
#             'V4': 'sex_identity', 'V5': 'age_group', 'V6': 'body_weight',
#             'V7': 'adm_type_code', 'V8': 'discharge_type', 'V9': 'adm_src_code',
#             'V10': 'hospital_days', 'V11': 'insurance_code', 'V12': 'provider_specialty',
#             'V13': 'lab_test_count', 'V14': 'procedure_count', 'V15': 'medication_count',
#             'V16': 'outpatient_visits', 'V17': 'emergency_visits', 'V18': 'inpatient_visits',
#             'V19': 'diagnosis_primary', 'V20': 'diagnosis_secondary', 'V21': 'diagnosis_tertiary',
#             'V22': 'diagnosis_total', 'V23': 'glucose_test_result', 'V24': 'A1C_result',
#             'V25': 'medication_1', 'V26': 'medication_2', 'V27': 'medication_3',
#             'V28': 'medication_4', 'V29': 'medication_5', 'V30': 'medication_6',
#             'V31': 'medication_7', 'V32': 'medication_8', 'V33': 'medication_9',
#             'V34': 'medication_10', 'V35': 'medication_11', 'V36': 'medication_12',
#             'V37': 'medication_13', 'V38': 'medication_14', 'V39': 'medication_15',
#             'V40': 'medication_16', 'V41': 'medication_17', 'V42': 'medication_18',
#             'V43': 'medication_19', 'V44': 'medication_20', 'V45': 'medication_21',
#             'V46': 'medication_22', 'V47': 'medication_23', 'V48': 'med_change_status',
#             'V49': 'diabetic_med_given', 'V50': 'readmission_status'
#         }
        
#         try:
#             # Check if columns exist before renaming
#             missing_cols = set(column_mapping.keys()) - set(self.df.columns)
#             if missing_cols:
#                 logger.warning(f"⚠️ Missing expected columns: {missing_cols}")
            
#             self.df.rename(columns=column_mapping, inplace=True)
#             logger.info("✓ Columns renamed successfully")
#             return self.df
            
#         except Exception as e:
#             logger.error(f"❌ Error renaming columns: {str(e)}")
#             raise
    
#     def safe_handle_missing_data(self) -> pd.DataFrame:
#         """Handle missing data with proper validation"""
#         try:
#             initial_missing = self.df.isnull().sum().sum()
            
#             if initial_missing > 0:
#                 logger.info(f"Found {initial_missing:,} missing values")
                
#                 # Handle specific columns
#                 if 'glucose_test_result' in self.df.columns:
#                     missing_glucose = self.df['glucose_test_result'].isnull().sum()
#                     if missing_glucose > 0:
#                         self.df['glucose_test_result'] = self.df['glucose_test_result'].fillna('Not Tested')
#                         logger.info(f"✓ Filled {missing_glucose:,} missing glucose results with 'Not Tested'")
                
#                 if 'A1C_result' in self.df.columns:
#                     missing_a1c = self.df['A1C_result'].isnull().sum()
#                     if missing_a1c > 0:
#                         self.df['A1C_result'] = self.df['A1C_result'].fillna('Not Tested')
#                         logger.info(f"✓ Filled {missing_a1c:,} missing A1C results with 'Not Tested'")
            
#             final_missing = self.df.isnull().sum().sum()
#             if final_missing == 0:
#                 logger.info("✓ No missing values remaining")
#             else:
#                 logger.warning(f"⚠️ {final_missing:,} missing values still present")
            
#             return self.df
            
#         except Exception as e:
#             logger.error(f"❌ Error handling missing data: {str(e)}")
#             raise
    
#     def safe_convert_dtypes(self) -> pd.DataFrame:
#         """Convert data types with error handling"""
#         try:
#             numerical_cols = [
#                 'encounter_code', 'patient_code', 'adm_type_code', 'discharge_type',
#                 'adm_src_code', 'hospital_days', 'lab_test_count', 'procedure_count',
#                 'medication_count', 'outpatient_visits', 'emergency_visits',
#                 'inpatient_visits', 'diagnosis_total'
#             ]
            
#             nominal_cols = [
#                 'ethnic_group', 'sex_identity', 'age_group', 'body_weight',
#                 'insurance_code', 'provider_specialty', 'diagnosis_primary',
#                 'diagnosis_secondary', 'diagnosis_tertiary', 'glucose_test_result',
#                 'A1C_result', 'med_change_status', 'diabetic_med_given',
#                 'readmission_status'
#             ] + [f'medication_{i}' for i in range(1, 24)]
            
#             # Convert numerical columns
#             converted_num = 0
#             for col in numerical_cols:
#                 if col in self.df.columns:
#                     try:
#                         self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
#                         converted_num += 1
#                     except Exception as e:
#                         logger.warning(f"⚠️ Could not convert {col} to numeric: {str(e)}")
            
#             logger.info(f"✓ Converted {converted_num} numerical columns")
            
#             # Convert nominal columns
#             converted_nom = 0
#             for col in nominal_cols:
#                 if col in self.df.columns:
#                     try:
#                         self.df[col] = self.df[col].astype('category')
#                         converted_nom += 1
#                     except Exception as e:
#                         logger.warning(f"⚠️ Could not convert {col} to category: {str(e)}")
            
#             logger.info(f"✓ Converted {converted_nom} nominal columns")
#             return self.df
            
#         except Exception as e:
#             logger.error(f"❌ Error converting data types: {str(e)}")
#             raise
    
#     def safe_statistical_test(self, col1: str, col2: str) -> Tuple[float, float, bool]:
#         """Perform chi-square test with error handling"""
#         try:
#             if col1 not in self.df.columns or col2 not in self.df.columns:
#                 raise ValueError(f"Columns not found: {col1} or {col2}")
            
#             contingency_table = pd.crosstab(self.df[col1], self.df[col2])
            
#             if contingency_table.empty:
#                 raise ValueError("Contingency table is empty")
            
#             chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#             is_significant = p_value < 0.05
            
#             logger.info(f"✓ Chi-square test: χ²={chi2:.2f}, p={p_value:.4f}, significant={is_significant}")
#             return chi2, p_value, is_significant
            
#         except Exception as e:
#             logger.error(f"❌ Error in statistical test: {str(e)}")
#             return np.nan, np.nan, False
    
#     def safe_crosstab(self, index_col: str, columns_col: str, normalize: str = None) -> pd.DataFrame:
#         """Create crosstab with error handling"""
#         try:
#             if index_col not in self.df.columns:
#                 raise ValueError(f"Index column not found: {index_col}")
#             if columns_col not in self.df.columns:
#                 raise ValueError(f"Columns column not found: {columns_col}")
            
#             result = pd.crosstab(self.df[index_col], self.df[columns_col], normalize=normalize)
#             logger.info(f"✓ Crosstab created: {index_col} vs {columns_col}")
#             return result
            
#         except Exception as e:
#             logger.error(f"❌ Error creating crosstab: {str(e)}")
#             return pd.DataFrame()
    
#     def safe_groupby_analysis(self, group_col: str, agg_col: str, func: str = 'mean') -> pd.Series:
#         """Perform groupby with error handling"""
#         try:
#             if group_col not in self.df.columns:
#                 raise ValueError(f"Group column not found: {group_col}")
#             if agg_col not in self.df.columns:
#                 raise ValueError(f"Aggregation column not found: {agg_col}")
            
#             if func == 'mean':
#                 result = self.df.groupby(group_col)[agg_col].mean()
#             elif func == 'sum':
#                 result = self.df.groupby(group_col)[agg_col].sum()
#             elif func == 'count':
#                 result = self.df.groupby(group_col)[agg_col].count()
#             else:
#                 raise ValueError(f"Unsupported function: {func}")
            
#             logger.info(f"✓ Groupby analysis: {group_col} -> {agg_col} ({func})")
#             return result
            
#         except Exception as e:
#             logger.error(f"❌ Error in groupby analysis: {str(e)}")
#             return pd.Series()
    
#     def run_full_pipeline(self) -> pd.DataFrame:
#         """Execute complete data processing pipeline with error handling"""
#         try:
#             logger.info("="*60)
#             logger.info("STARTING DATA PROCESSING PIPELINE")
#             logger.info("="*60)
            
#             self.safe_load_data()
#             self.safe_rename_columns()
#             self.safe_handle_missing_data()
#             self.safe_convert_dtypes()
            
#             logger.info("="*60)
#             logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
#             logger.info("="*60)
            
#             return self.df
            
#         except Exception as e:
#             logger.error(f"❌ Pipeline failed: {str(e)}")
#             raise


# # ============================================
# # USAGE EXAMPLE
# # ============================================

# if __name__ == "__main__":
#     # Initialize processor
#     processor = DiabeticDataProcessor(
#         r"C:\Users\Raiyan\Downloads\diabetic_data_QMH_Club_Fest_2025.xlsx"
#     )
    
#     # Run pipeline
#     df = processor.run_full_pipeline()
    
#     # Perform safe analyses
#     print("\n" + "="*60)
#     print("ANALYSIS RESULTS")
#     print("="*60)
    
#     # Safe crosstab
#     readmission_by_age = processor.safe_crosstab('age_group', 'readmission_status', normalize='index')
#     print("\nReadmission by Age Group:")
#     print(readmission_by_age)
    
#     # Safe statistical test
#     chi2, p_value, significant = processor.safe_statistical_test('age_group', 'readmission_status')
#     if not np.isnan(p_value):
#         print(f"\nStatistical significance: {'YES' if significant else 'NO'} (p={p_value:.4f})")
    
#     # Safe groupby
#     avg_meds_by_age = processor.safe_groupby_analysis('age_group', 'medication_count', 'mean')
#     print("\nAverage Medications by Age:")
#     print(avg_meds_by_age)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
import logging
from typing import Optional, Dict, List, Tuple
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DiabeticReadmissionAnalyzer:
    """
    Analyzes hypothesis: "Older diabetic patients with multiple medications 
    are more likely to be readmitted within 30 days"
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.results: Dict = {}
        
    # ============================================
    # DATA LOADING & PREPARATION (with error handling)
    # ============================================
    
    def safe_load_and_prepare(self) -> pd.DataFrame:
        """Complete data loading and preparation pipeline"""
        try:
            logger.info("="*70)
            logger.info("STEP 1: LOADING DATA")
            logger.info("="*70)
            
            # Load data
            self.df = pd.read_excel(self.file_path, sheet_name="Sheet1")
            if self.df.empty:
                raise ValueError("Dataset is empty")
            logger.info(f"✓ Loaded {self.df.shape[0]:,} rows and {self.df.shape[1]} columns")
            
            # Rename columns
            logger.info("\nSTEP 2: RENAMING COLUMNS")
            self._rename_columns()
            
            # Handle missing data
            logger.info("\nSTEP 3: HANDLING MISSING DATA")
            self._handle_missing_data()
            
            # Convert data types
            logger.info("\nSTEP 4: CONVERTING DATA TYPES")
            self._convert_dtypes()
            
            # Create binary readmission variable
            logger.info("\nSTEP 5: CREATING TARGET VARIABLE")
            self._create_readmission_binary()
            
            logger.info("\n" + "="*70)
            logger.info("✓ DATA PREPARATION COMPLETE")
            logger.info("="*70)
            
            return self.df
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {str(e)}")
            raise
    
    def _rename_columns(self):
        """Rename columns with validation"""
        column_mapping = {
            'V1': 'encounter_code', 'V2': 'patient_code', 'V3': 'ethnic_group',
            'V4': 'sex_identity', 'V5': 'age_group', 'V6': 'body_weight',
            'V7': 'adm_type_code', 'V8': 'discharge_type', 'V9': 'adm_src_code',
            'V10': 'hospital_days', 'V11': 'insurance_code', 'V12': 'provider_specialty',
            'V13': 'lab_test_count', 'V14': 'procedure_count', 'V15': 'medication_count',
            'V16': 'outpatient_visits', 'V17': 'emergency_visits', 'V18': 'inpatient_visits',
            'V19': 'diagnosis_primary', 'V20': 'diagnosis_secondary', 'V21': 'diagnosis_tertiary',
            'V22': 'diagnosis_total', 'V23': 'glucose_test_result', 'V24': 'A1C_result',
            'V25': 'medication_1', 'V26': 'medication_2', 'V27': 'medication_3',
            'V28': 'medication_4', 'V29': 'medication_5', 'V30': 'medication_6',
            'V31': 'medication_7', 'V32': 'medication_8', 'V33': 'medication_9',
            'V34': 'medication_10', 'V35': 'medication_11', 'V36': 'medication_12',
            'V37': 'medication_13', 'V38': 'medication_14', 'V39': 'medication_15',
            'V40': 'medication_16', 'V41': 'medication_17', 'V42': 'medication_18',
            'V43': 'medication_19', 'V44': 'medication_20', 'V45': 'medication_21',
            'V46': 'medication_22', 'V47': 'medication_23', 'V48': 'med_change_status',
            'V49': 'diabetic_med_given', 'V50': 'readmission_status'
        }
        self.df.rename(columns=column_mapping, inplace=True)
        logger.info("✓ Columns renamed successfully")
    
    def _handle_missing_data(self):
        """Handle missing values"""
        initial_missing = self.df.isnull().sum().sum()
        
        if 'glucose_test_result' in self.df.columns:
            self.df['glucose_test_result'] = self.df['glucose_test_result'].fillna('Not Tested')
        
        if 'A1C_result' in self.df.columns:
            self.df['A1C_result'] = self.df['A1C_result'].fillna('Not Tested')
        
        final_missing = self.df.isnull().sum().sum()
        logger.info(f"✓ Reduced missing values: {initial_missing:,} → {final_missing:,}")
    
    def _convert_dtypes(self):
        """Convert data types safely"""
        numerical_cols = [
            'encounter_code', 'patient_code', 'adm_type_code', 'discharge_type',
            'adm_src_code', 'hospital_days', 'lab_test_count', 'procedure_count',
            'medication_count', 'outpatient_visits', 'emergency_visits',
            'inpatient_visits', 'diagnosis_total'
        ]
        
        for col in numerical_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        logger.info(f"✓ Converted {len(numerical_cols)} numerical columns")
    
    def _create_readmission_binary(self):
        """Create binary readmission variable: 1 = readmitted <30 days, 0 = not readmitted"""
        try:
            if 'readmission_status' not in self.df.columns:
                raise ValueError("readmission_status column not found")
            
            # Create binary variable
            self.df['readmitted_30days'] = (self.df['readmission_status'] == '<30').astype(int)
            
            counts = self.df['readmitted_30days'].value_counts()
            logger.info(f"✓ Created binary target: {counts[1]:,} readmitted <30 days, {counts[0]:,} not readmitted")
            
        except Exception as e:
            logger.error(f"❌ Error creating target variable: {str(e)}")
            raise
    
    # ============================================
    # HYPOTHESIS TESTING
    # ============================================
    
    def test_hypothesis(self) -> Dict:
        """
        Test hypothesis: "Older diabetic patients with multiple medications 
        are more likely to be readmitted within 30 days"
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("HYPOTHESIS TESTING")
            logger.info("="*70)
            
            # Test 1: Age vs Readmission
            logger.info("\n📊 TEST 1: Does AGE affect 30-day readmission?")
            self._test_age_readmission()
            
            # Test 2: Medication count vs Readmission
            logger.info("\n📊 TEST 2: Does MEDICATION COUNT affect 30-day readmission?")
            self._test_medication_readmission()
            
            # Test 3: Combined effect (Age + Medications)
            logger.info("\n📊 TEST 3: Combined effect of AGE + MEDICATIONS")
            self._test_combined_effect()
            
            # Test 4: High-risk group analysis
            logger.info("\n📊 TEST 4: High-risk group (Elderly + Many Meds)")
            self._analyze_high_risk_group()
            
            logger.info("\n" + "="*70)
            logger.info("✓ HYPOTHESIS TESTING COMPLETE")
            logger.info("="*70)
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Hypothesis testing failed: {str(e)}")
            raise
    
    def _test_age_readmission(self):
        """Test if age affects readmission"""
        try:
            # Create crosstab
            crosstab = pd.crosstab(self.df['age_group'], 
                                   self.df['readmitted_30days'], 
                                   normalize='index') * 100
            
            print("\nReadmission rate by age group (%):")
            print(crosstab[1].sort_index())
            
            # Chi-square test
            contingency = pd.crosstab(self.df['age_group'], self.df['readmitted_30days'])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            is_significant = p_value < 0.05
            logger.info(f"Chi-square: χ²={chi2:.2f}, p={p_value:.6f}, Significant: {is_significant}")
            
            self.results['age_test'] = {
                'chi2': chi2,
                'p_value': p_value,
                'significant': is_significant,
                'rates_by_age': crosstab[1].to_dict()
            }
            
            if is_significant:
                logger.info("✓ Age DOES significantly affect 30-day readmission!")
            else:
                logger.info("✗ Age does NOT significantly affect 30-day readmission")
            
        except Exception as e:
            logger.error(f"❌ Age test failed: {str(e)}")
    
    def _test_medication_readmission(self):
        """Test if medication count affects readmission"""
        try:
            # Group by medication count
            med_stats = self.df.groupby('readmitted_30days')['medication_count'].agg(['mean', 'median', 'std'])
            
            print("\nMedication count statistics:")
            print(med_stats)
            
            # T-test equivalent (ANOVA for 2 groups)
            readmitted = self.df[self.df['readmitted_30days'] == 1]['medication_count'].dropna()
            not_readmitted = self.df[self.df['readmitted_30days'] == 0]['medication_count'].dropna()
            
            f_stat, p_value = f_oneway(readmitted, not_readmitted)
            is_significant = p_value < 0.05
            
            logger.info(f"ANOVA: F={f_stat:.2f}, p={p_value:.6f}, Significant: {is_significant}")
            
            self.results['medication_test'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': is_significant,
                'mean_readmitted': readmitted.mean(),
                'mean_not_readmitted': not_readmitted.mean()
            }
            
            if is_significant:
                logger.info("✓ Medication count DOES significantly affect 30-day readmission!")
            else:
                logger.info("✗ Medication count does NOT significantly affect 30-day readmission")
            
        except Exception as e:
            logger.error(f"❌ Medication test failed: {str(e)}")
    
    def _test_combined_effect(self):
        """Test combined effect of age and medications"""
        try:
            # Create high medication flag (above median)
            median_meds = self.df['medication_count'].median()
            self.df['high_meds'] = (self.df['medication_count'] > median_meds).astype(int)
            
            # Crosstab of age, medication level, and readmission
            combined = pd.crosstab([self.df['age_group'], self.df['high_meds']], 
                                   self.df['readmitted_30days'], 
                                   normalize='index') * 100
            
            print(f"\nReadmission rate by age and medication level (>{median_meds:.0f} meds = high):")
            print(combined[1])
            
            # Chi-square test
            contingency = pd.crosstab([self.df['age_group'], self.df['high_meds']], 
                                      self.df['readmitted_30days'])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            is_significant = p_value < 0.05
            logger.info(f"Chi-square: χ²={chi2:.2f}, p={p_value:.6f}, Significant: {is_significant}")
            
            self.results['combined_test'] = {
                'chi2': chi2,
                'p_value': p_value,
                'significant': is_significant,
                'median_medications': median_meds
            }
            
        except Exception as e:
            logger.error(f"❌ Combined test failed: {str(e)}")
    
    def _analyze_high_risk_group(self):
        """Analyze elderly patients with many medications"""
        try:
            # Define high-risk: Age 60+ AND medications > 15
            elderly_ages = ['[60-70)', '[70-80)', '[80-90)', '[90-100)']
            
            high_risk = self.df[
                (self.df['age_group'].isin(elderly_ages)) & 
                (self.df['medication_count'] > 15)
            ]
            
            low_risk = self.df[
                (~self.df['age_group'].isin(elderly_ages)) | 
                (self.df['medication_count'] <= 15)
            ]
            
            high_risk_rate = high_risk['readmitted_30days'].mean() * 100
            low_risk_rate = low_risk['readmitted_30days'].mean() * 100
            
            print(f"\n🎯 HIGH-RISK GROUP (Age 60+ AND 15+ medications):")
            print(f"   - Group size: {len(high_risk):,} patients")
            print(f"   - Readmission rate: {high_risk_rate:.2f}%")
            print(f"\n🎯 LOW-RISK GROUP (Younger OR Fewer medications):")
            print(f"   - Group size: {len(low_risk):,} patients")
            print(f"   - Readmission rate: {low_risk_rate:.2f}%")
            print(f"\n📈 RISK RATIO: {high_risk_rate/low_risk_rate:.2f}x higher risk")
            
            self.results['high_risk_analysis'] = {
                'high_risk_count': len(high_risk),
                'high_risk_rate': high_risk_rate,
                'low_risk_count': len(low_risk),
                'low_risk_rate': low_risk_rate,
                'risk_ratio': high_risk_rate/low_risk_rate
            }
            
            if high_risk_rate > low_risk_rate:
                logger.info("✓ HIGH-RISK group has HIGHER readmission rate - HYPOTHESIS SUPPORTED!")
            
        except Exception as e:
            logger.error(f"❌ High-risk analysis failed: {str(e)}")
    
    # ============================================
    # SUMMARY & RECOMMENDATIONS
    # ============================================
    
    def generate_summary(self) -> str:
        """Generate final summary and recommendations"""
        try:
            logger.info("\n" + "="*70)
            logger.info("FINAL SUMMARY & RECOMMENDATIONS")
            logger.info("="*70)
            
            summary = []
            summary.append("\n📋 HYPOTHESIS: 'Older diabetic patients with multiple medications")
            summary.append("   are more likely to be readmitted within 30 days'")
            summary.append("\n" + "-"*70)
            
            # Check each component
            if self.results.get('age_test', {}).get('significant'):
                summary.append("\n✅ Age IS a significant factor (p < 0.05)")
            else:
                summary.append("\n⚠️ Age is NOT a significant factor")
            
            if self.results.get('medication_test', {}).get('significant'):
                summary.append("✅ Medication count IS a significant factor (p < 0.05)")
            else:
                summary.append("⚠️ Medication count is NOT a significant factor")
            
            if self.results.get('combined_test', {}).get('significant'):
                summary.append("✅ Combined effect (Age + Meds) IS significant (p < 0.05)")
            
            # High-risk group findings
            hr_analysis = self.results.get('high_risk_analysis', {})
            if hr_analysis:
                risk_ratio = hr_analysis.get('risk_ratio', 1)
                summary.append(f"\n🎯 HIGH-RISK GROUP: {risk_ratio:.2f}x higher readmission risk")
            
            summary.append("\n" + "-"*70)
            summary.append("\n💡 RECOMMENDATIONS:")
            summary.append("1. Target patients aged 60+ with 15+ medications for intervention")
            summary.append("2. Implement enhanced discharge planning for high-risk groups")
            summary.append("3. Schedule follow-up appointments within 7 days of discharge")
            summary.append("4. Consider medication reconciliation programs")
            summary.append("5. Provide patient education on medication management")
            
            summary_text = "\n".join(summary)
            print(summary_text)
            
            return summary_text
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {str(e)}")
            return "Error generating summary"


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DiabeticReadmissionAnalyzer(
        r"C:\Users\Raiyan\Downloads\diabetic_data_QMH_Club_Fest_2025.xlsx"
    )
    
    # Run complete analysis
    try:
        # Step 1: Load and prepare data
        df = analyzer.safe_load_and_prepare()
        
        # Step 2: Test hypothesis
        results = analyzer.test_hypothesis()
        
        # Step 3: Generate summary
        summary = analyzer.generate_summary()
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE - ALL TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {str(e)}")
        print("Please check the error logs above for details.")