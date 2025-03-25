import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any
import joblib
from datetime import datetime

class RiskScorer:
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.feature_columns = [
            'account_balance',
            'transaction_amount',
            'reported_amount'
        ]
        self.risk_factors = {
            'cross_border': 2.0,
            'large_amount': 1.5,
            'currency_mismatch': 1.8,
            'historical_violations': 2.5
        }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for risk scoring."""
        features = df[self.feature_columns].copy()
        
        # Add derived features
        features['amount_ratio'] = df['transaction_amount'] / df['account_balance']
        features['is_cross_border'] = (df['country'] != 'US').astype(float)
        features['has_currency_mismatch'] = (
            (df['transaction_amount'] != df['reported_amount']).astype(float)
        )
        
        return features

    def train(self, df: pd.DataFrame):
        """Train the anomaly detection model."""
        features = self.prepare_features(df)
        self.model.fit(features)

    def calculate_base_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate base risk scores using the trained model."""
        features = self.prepare_features(df)
        # Convert isolation forest decision scores to risk scores (0-10 scale)
        raw_scores = -self.model.score_samples(features)
        min_score, max_score = raw_scores.min(), raw_scores.max()
        normalized_scores = (raw_scores - min_score) / (max_score - min_score)
        return normalized_scores * 10

    def apply_risk_factors(self, df: pd.DataFrame, base_scores: np.ndarray) -> np.ndarray:
        """Apply risk factors to base scores."""
        risk_multiplier = np.ones(len(df))
        
        # Cross-border transactions
        risk_multiplier[df['country'] != 'US'] *= self.risk_factors['cross_border']
        
        # Large transactions
        large_amount_mask = df['transaction_amount'] > df['transaction_amount'].mean() * 2
        risk_multiplier[large_amount_mask] *= self.risk_factors['large_amount']
        
        # Currency mismatches
        currency_mismatch_mask = df['transaction_amount'] != df['reported_amount']
        risk_multiplier[currency_mismatch_mask] *= self.risk_factors['currency_mismatch']
        
        return np.clip(base_scores * risk_multiplier, 0, 10)

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score transactions and return DataFrame with risk scores."""
        # Calculate base risk scores
        base_scores = self.calculate_base_risk_score(df)
        
        # Apply risk factors
        final_scores = self.apply_risk_factors(df, base_scores)
        
        # Add scores to DataFrame
        result_df = df.copy()
        result_df['calculated_risk_score'] = final_scores
        result_df['risk_level'] = pd.cut(
            final_scores,
            bins=[0, 3, 7, 10],
            labels=['Low', 'Medium', 'High']
        )
        
        return result_df

    def save_model(self, path: str):
        """Save the trained model to disk."""
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load a trained model from disk."""
        self.model = joblib.load(path)

    def get_risk_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights about risk patterns."""
        scored_df = self.score_transactions(df)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_transactions": len(df),
            "risk_distribution": {
                "high": len(scored_df[scored_df['risk_level'] == 'High']),
                "medium": len(scored_df[scored_df['risk_level'] == 'Medium']),
                "low": len(scored_df[scored_df['risk_level'] == 'Low'])
            },
            "average_risk_score": scored_df['calculated_risk_score'].mean(),
            "high_risk_factors": {
                "cross_border": len(scored_df[
                    (scored_df['risk_level'] == 'High') & 
                    (scored_df['country'] != 'US')
                ]),
                "large_amount": len(scored_df[
                    (scored_df['risk_level'] == 'High') & 
                    (scored_df['transaction_amount'] > scored_df['transaction_amount'].mean() * 2)
                ]),
                "currency_mismatch": len(scored_df[
                    (scored_df['risk_level'] == 'High') & 
                    (scored_df['transaction_amount'] != scored_df['reported_amount'])
                ])
            }
        }
