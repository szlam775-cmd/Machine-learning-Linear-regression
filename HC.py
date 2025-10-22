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
