

"""
Student Grade Prediction ML System
Predicts student grades based on study hours using Linear Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class GradePredictionModel:
    """
    A machine learning model to predict student grades based on study hours.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.metrics = {}
        
    def load_data(self, filepath=None, data=None):
        """
        Load data from CSV file or use provided data.
        
        Args:
            filepath: Path to CSV file with columns 'study_hours' and 'grade'
            data: Pandas DataFrame (optional, if not loading from file)
        
        Returns:
            DataFrame with study_hours and grade columns
        """
        if filepath:
            df = pd.read_csv(filepath)
        elif data is not None:
            df = data
        else:
            # Generate sample data if no data provided
            print("No data provided. Generating sample dataset...")
            df = self.generate_sample_data()
        
        # Validate required columns
        if 'study_hours' not in df.columns or 'grade' not in df.columns:
            raise ValueError("Data must contain 'study_hours' and 'grade' columns")
        
        return df
    
    def generate_sample_data(self, n_samples=100):
        """
        Generate sample dataset for demonstration.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            DataFrame with study_hours and grade columns
        """
        np.random.seed(42)
        study_hours = np.random.uniform(1, 40, n_samples)
        # Grade = base + study_effect + noise
        grades = 30 + (1.5 * study_hours) + np.random.normal(0, 8, n_samples)
        grades = np.clip(grades, 0, 100)  # Keep grades in valid range
        
        df = pd.DataFrame({
            'study_hours': study_hours,
            'grade': grades
        })
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data: handle missing values and outliers.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        # Handle missing values
        df = df.dropna()
        
        # Remove outliers using IQR method
        Q1 = df['study_hours'].quantile(0.25)
        Q3 = df['study_hours'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df['study_hours'] >= lower_bound) & (df['study_hours'] <= upper_bound)]
        
        # Ensure grades are in valid range
        df = df[(df['grade'] >= 0) & (df['grade'] <= 100)]
        
        return df
    
    def train(self, df, test_size=0.2, random_state=42):
        """
        Train the model on the provided data.
        
        Args:
            df: DataFrame with study_hours and grade columns
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Split features and target
        X = df[['study_hours']].values
        y = df['grade'].values
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        
        # Calculate metrics
        self._calculate_metrics()
        
        print("Model trained successfully!")
        print(f"\nModel Performance:")
        print(f"R² Score: {self.metrics['r2_score']:.4f}")
        print(f"RMSE: {self.metrics['rmse']:.4f}")
        print(f"MAE: {self.metrics['mae']:.4f}")
        print(f"\nModel Equation: Grade = {self.model.intercept_:.2f} + {self.model.coef_[0]:.2f} × Study Hours")
        
    def _calculate_metrics(self):
        """Calculate and store model performance metrics."""
        y_pred = self.model.predict(self.X_test)
        
        self.metrics = {
            'r2_score': r2_score(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'mse': mean_squared_error(self.y_test, y_pred)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                    cv=5, scoring='r2')
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()
    
    def predict(self, study_hours):
        """
        Predict grade for given study hours.
        
        Args:
            study_hours: Number of hours studied (float or list)
        
        Returns:
            Predicted grade(s)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle single value or list
        if isinstance(study_hours, (int, float)):
            study_hours = [[study_hours]]
        elif isinstance(study_hours, list):
            study_hours = [[h] for h in study_hours]
        
        prediction = self.model.predict(study_hours)
        return np.clip(prediction, 0, 100)  # Ensure grades are in valid range
    
    def visualize_results(self):
        """Create visualizations of model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Actual vs Predicted (Training Data)
        y_train_pred = self.model.predict(self.X_train)
        axes[0, 0].scatter(self.X_train, self.y_train, alpha=0.6, label='Actual')
        axes[0, 0].plot(self.X_train, y_train_pred, 'r-', linewidth=2, label='Predicted')
        axes[0, 0].set_xlabel('Study Hours')
        axes[0, 0].set_ylabel('Grade')
        axes[0, 0].set_title('Training Data: Actual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Test Set Performance
        y_test_pred = self.model.predict(self.X_test)
        axes[0, 1].scatter(self.X_test, self.y_test, alpha=0.6, label='Actual')
        axes[0, 1].scatter(self.X_test, y_test_pred, alpha=0.6, color='red', label='Predicted')
        axes[0, 1].set_xlabel('Study Hours')
        axes[0, 1].set_ylabel('Grade')
        axes[0, 1].set_title('Test Data: Actual vs Predicted')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Residual Plot
        residuals = self.y_test - y_test_pred
        axes[1, 0].scatter(y_test_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Predicted Grade')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residual Plot')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Prediction vs Actual
        axes[1, 1].scatter(self.y_test, y_test_pred, alpha=0.6)
        axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 1].set_xlabel('Actual Grade')
        axes[1, 1].set_ylabel('Predicted Grade')
        axes[1, 1].set_title(f'Predicted vs Actual (R²={self.metrics["r2_score"]:.3f})')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='grade_prediction_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where model will be saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='grade_prediction_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.metrics = model_data['metrics']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize model
    gpm = GradePredictionModel()
    
    # Load or generate data
    print("Loading data...")
    df = gpm.load_data()  # Will generate sample data
    print(f"Dataset size: {len(df)} samples")
    print(f"\nData preview:\n{df.head()}")
    
    # Train model
    print("\n" + "="*50)
    print("Training model...")
    print("="*50)
    gpm.train(df)
    
    # Make predictions
    print("\n" + "="*50)
    print("Making predictions...")
    print("="*50)
    
    test_hours = [5, 10, 15, 20, 25, 30]
    for hours in test_hours:
        predicted_grade = gpm.predict(hours)[0]
        print(f"Study Hours: {hours:2d} → Predicted Grade: {predicted_grade:.2f}")
    
    # Visualize results
    print("\n" + "="*50)
    print("Generating visualizations...")
    print("="*50)
    gpm.visualize_results()
    
    # Save model
    print("\n" + "="*50)
    gpm.save_model()
    print("="*50)
    
    print("\n✓ Model training and evaluation complete!")
    print("\nTo use this model:")
    print("1. Load your data: df = gpm.load_data('study_hours_grades(4).csv')")
    print("2. Train: gpm.train(df)")
    print("3. Predict: gpm.predict(study_hours)")

