import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
import joblib
from datetime import datetime

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = [
            'account_balance',
            'transaction_amount',
            'reported_amount'
        ]

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare and scale features for anomaly detection."""
        features = df[self.feature_columns].copy()
        
        # Add derived features
        features['amount_ratio'] = df['transaction_amount'] / df['account_balance']
        features['amount_difference'] = abs(
            df['transaction_amount'] - df['reported_amount']
        )
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features

    def train(self, df: pd.DataFrame):
        """Train the anomaly detection model."""
        features = self.prepare_features(df)
        self.isolation_forest.fit(features)

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the dataset."""
        features = self.prepare_features(df)
        
        # Get anomaly scores (-1 for anomalies, 1 for normal)
        anomaly_labels = self.isolation_forest.predict(features)
        anomaly_scores = -self.isolation_forest.score_samples(features)
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df['is_anomaly'] = anomaly_labels == -1
        result_df['anomaly_score'] = anomaly_scores
        
        return result_df

    def get_anomaly_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights about detected anomalies."""
        anomaly_df = self.detect_anomalies(df)
        anomalies = anomaly_df[anomaly_df['is_anomaly']]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_transactions": len(df),
            "total_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / len(df) * 100,
            "average_anomaly_score": anomalies['anomaly_score'].mean(),
            "top_anomalies": self._get_top_anomalies(anomalies),
            "anomaly_patterns": self._analyze_anomaly_patterns(anomalies)
        }

    def _get_top_anomalies(self, anomalies: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get details of top N anomalies by score."""
        top_anomalies = anomalies.nlargest(top_n, 'anomaly_score')
        
        return [{
            "transaction_id": idx,
            "customer_id": row['customerid'],
            "amount": row['transaction_amount'],
            "reported_amount": row['reported_amount'],
            "currency": row['currency'],
            "country": row['country'],
            "anomaly_score": row['anomaly_score'],
            "reason": self._get_anomaly_reason(row)
        } for idx, row in top_anomalies.iterrows()]

    def _analyze_anomaly_patterns(self, anomalies: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in anomalous transactions."""
        return {
            "currency_distribution": anomalies['currency'].value_counts().to_dict(),
            "country_distribution": anomalies['country'].value_counts().to_dict(),
            "average_transaction_amount": anomalies['transaction_amount'].mean(),
            "amount_std_dev": anomalies['transaction_amount'].std(),
            "common_patterns": self._identify_common_patterns(anomalies)
        }

    def _identify_common_patterns(self, anomalies: pd.DataFrame) -> List[str]:
        """Identify common patterns in anomalous transactions."""
        patterns = []
        
        # Check for large amount discrepancies
        if (abs(anomalies['transaction_amount'] - anomalies['reported_amount']) > 
            anomalies['transaction_amount'].mean() * 0.1).any():
            patterns.append("Large amount discrepancies detected")
        
        # Check for unusual account balance ratios
        if (anomalies['transaction_amount'] / anomalies['account_balance'] > 0.5).any():
            patterns.append("High transaction to balance ratios")
        
        # Check for cross-border patterns
        if (anomalies['country'] != 'US').any():
            patterns.append("Unusual cross-border transaction patterns")
        
        return patterns

    def _get_anomaly_reason(self, transaction: pd.Series) -> str:
        """Generate explanation for why a transaction is anomalous."""
        reasons = []
        
        # Check amount discrepancy
        if transaction['transaction_amount'] != transaction['reported_amount']:
            reasons.append("Amount mismatch")
        
        # Check transaction size relative to balance
        if transaction['transaction_amount'] > transaction['account_balance'] * 0.5:
            reasons.append("Large transaction relative to balance")
        
        # Check cross-border
        if transaction['country'] != 'US':
            reasons.append("Cross-border transaction")
        
        return ", ".join(reasons) if reasons else "Multiple unusual patterns detected"

    def save_model(self, path: str):
        """Save the trained model to disk."""
        joblib.dump({
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler
        }, path)

    def load_model(self, path: str):
        """Load a trained model from disk."""
        models = joblib.load(path)
        self.isolation_forest = models['isolation_forest']
        self.scaler = models['scaler']
