import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from src.utils import (
    is_valid_currency,
    is_valid_country,
    calculate_amount_deviation,
    format_validation_result,
    get_high_risk_threshold,
    is_cross_border_transaction
)

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_results = []

    def validate_all(self) -> List[Dict[str, Any]]:
        """Run all validation checks on the dataset."""
        self.validate_amounts()
        self.validate_account_balances()
        self.validate_currencies()
        self.validate_countries()
        self.validate_dates()
        self.validate_high_risk_transactions()
        return self.validation_results

    def validate_amounts(self):
        """Validate transaction amounts match reported amounts."""
        for idx, row in self.df.iterrows():
            deviation = calculate_amount_deviation(
                row['transaction_amount'],
                row['reported_amount']
            )
            
            # Allow 1% deviation for cross-currency transactions
            max_allowed_deviation = 1.0 if row['currency'] != 'USD' else 0.0
            
            if deviation > max_allowed_deviation:
                self.validation_results.append(
                    format_validation_result(
                        "amount_match",
                        False,
                        f"Transaction {idx}: Amount deviation of {deviation:.2f}% exceeds allowed {max_allowed_deviation}%"
                    )
                )

    def validate_account_balances(self):
        """Validate account balances are not negative."""
        negative_balances = self.df[self.df['account_balance'] < 0]
        for idx, row in negative_balances.iterrows():
            self.validation_results.append(
                format_validation_result(
                    "account_balance",
                    False,
                    f"Account {row['customerid']}: Negative balance of {row['account_balance']}"
                )
            )

    def validate_currencies(self):
        """Validate currency codes against ISO 4217."""
        for idx, row in self.df.iterrows():
            if not is_valid_currency(row['currency']):
                self.validation_results.append(
                    format_validation_result(
                        "currency",
                        False,
                        f"Transaction {idx}: Invalid currency code {row['currency']}"
                    )
                )

    def validate_countries(self):
        """Validate country codes and cross-border transaction rules."""
        threshold = get_high_risk_threshold()
        
        for idx, row in self.df.iterrows():
            if not is_valid_country(row['country']):
                self.validation_results.append(
                    format_validation_result(
                        "country",
                        False,
                        f"Transaction {idx}: Invalid country code {row['country']}"
                    )
                )
            
            # Check cross-border transaction rules
            if (is_cross_border_transaction(row) and 
                row['transaction_amount'] > threshold):
                self.validation_results.append(
                    format_validation_result(
                        "cross_border",
                        False,
                        f"Transaction {idx}: High-value cross-border transaction requires additional documentation"
                    )
                )

    def validate_dates(self):
        """Validate transaction dates are not in the future."""
        current_date = datetime.now()
        future_transactions = self.df[self.df['transaction_date'] > current_date]
        
        for idx, row in future_transactions.iterrows():
            self.validation_results.append(
                format_validation_result(
                    "transaction_date",
                    False,
                    f"Transaction {idx}: Future date detected {row['transaction_date']}"
                )
            )

    def validate_high_risk_transactions(self):
        """Flag high-risk transactions for compliance check."""
        high_risk_transactions = self.df[self.df['risk_score'] >= 7]
        
        for idx, row in high_risk_transactions.iterrows():
            self.validation_results.append(
                format_validation_result(
                    "risk_score",
                    False,
                    f"Transaction {idx}: High risk score {row['risk_score']} requires compliance review"
                )
            )

    def get_validation_summary(self) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        total_issues = len(self.validation_results)
        issues_by_type = {}
        
        for result in self.validation_results:
            issue_type = result['type']
            issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
        
        return {
            "total_transactions": len(self.df),
            "total_issues": total_issues,
            "issues_by_type": issues_by_type,
            "validation_timestamp": datetime.now().isoformat()
        }
