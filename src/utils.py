import pandas as pd
import pycountry
from datetime import datetime
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    try:
        df = pd.read_csv(file_path)
        required_columns = [
            'customerid', 'account_balance', 'transaction_amount',
            'reported_amount', 'currency', 'country', 'transaction_date',
            'risk_score'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert date column to datetime
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def is_valid_currency(currency: str) -> bool:
    """Check if the currency code is valid according to ISO 4217."""
    try:
        return bool(pycountry.currencies.get(alpha_3=currency))
    except:
        return False

def is_valid_country(country: str) -> bool:
    """Check if the country code is valid."""
    try:
        return bool(pycountry.countries.get(alpha_2=country))
    except:
        return False

def calculate_amount_deviation(amount1: float, amount2: float) -> float:
    """Calculate the percentage deviation between two amounts."""
    if amount1 == 0 and amount2 == 0:
        return 0
    if amount1 == 0:
        return float('inf')
    return abs((amount1 - amount2) / amount1) * 100

def format_validation_result(validation_type: str, status: bool, details: str) -> Dict[str, Any]:
    """Format validation results in a consistent structure."""
    return {
        "type": validation_type,
        "passed": status,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }

def get_high_risk_threshold() -> float:
    """Get the threshold for high-risk transactions."""
    return 25000  # Default threshold for high-risk transactions

def is_cross_border_transaction(row: pd.Series) -> bool:
    """Determine if a transaction is cross-border based on country codes."""
    return row['country'] != 'US'  # Assuming US is the home country

def format_currency_amount(amount: float, currency: str) -> str:
    """Format currency amounts with proper currency symbols."""
    return f"{amount:,.2f} {currency}"
