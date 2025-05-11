from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np

class ChurnPredictor(BaseEstimator):
    """Advanced Churn Prediction Model with CLTV calculation"""
    
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.feature_importance_ = None
        self.scaler = StandardScaler()
        
        # Model selection
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)
        ])
        
    def fit(self, X, y):
        """Train the model"""
        self.pipeline.fit(X, y)
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        return self
    
    def predict(self, X):
        """Predict churn probability"""
        return self.pipeline.predict_proba(X)[:, 1]
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = (self.predict(X) > 0.5).astype(int)
        return {
            'classification_report': classification_report(y, y_pred),
            'roc_auc': roc_auc_score(y, self.predict(X))
        }
    
    def calculate_cltv(self, df, tenure_col='tenure', 
                      charges_col='MonthlyCharges',
                      discount_rate=0.1):
        """
        Calculate Customer Lifetime Value
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        tenure_col : str
            Column name for customer tenure
        charges_col : str
            Column name for monthly charges
        discount_rate : float
            Annual discount rate for future value calculation
        """
        # Predict churn probability
        churn_prob = self.predict(df)
        
        # Calculate expected lifetime in months
        monthly_discount = (1 + discount_rate)**(1/12) - 1
        expected_lifetime = 1 / (churn_prob + monthly_discount)
        
        # Calculate CLTV
        monthly_value = df[charges_col]
        cltv = monthly_value * expected_lifetime
        
        return cltv
    
    def segment_customers(self, df, cltv):
        """
        Segment customers based on CLTV and churn risk
        
        Returns dictionary with customer segments and recommendations
        """
        churn_prob = self.predict(df)
        segments = pd.DataFrame({
            'cltv': cltv,
            'churn_prob': churn_prob
        })
        
        # Define segments
        segments['segment'] = pd.qcut(segments['cltv'], q=3, labels=['Low', 'Medium', 'High'])
        segments['risk_level'] = pd.qcut(segments['churn_prob'], q=3, labels=['Low', 'Medium', 'High'])
        
        return segments

def get_recommendations(segment, risk_level):
    """Generate recommendations based on segment and risk level"""
    recommendations = {
        'High': {
            'High': 'Priority retention: Offer premium retention package',
            'Medium': 'Proactive engagement: Upgrade services and benefits',
            'Low': 'Growth focus: Upsell premium services'
        },
        'Medium': {
            'High': 'Retention focus: Offer competitive upgrades',
            'Medium': 'Satisfaction monitoring: Regular check-ins',
            'Low': 'Value enhancement: Suggest service additions'
        },
        'Low': {
            'High': 'Basic retention: Offer essential benefits',
            'Medium': 'Service improvement: Address pain points',
            'Low': 'Maintain relationship: Regular communications'
        }
    }
    return recommendations[segment][risk_level]