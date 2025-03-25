# Regulatory Compliance Assistant

An interactive compliance assistant that helps auditors validate and profile financial transaction data using AI/ML techniques.

## Features

- Interactive compliance assistant with conversational interface
- Automated data profiling and validation
- Anomaly detection for transactions
- Risk scoring system
- Remediation suggestions for flagged transactions

## Setup Instructions

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On Windows:
```bash
.\venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a .env file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

6. Run the application:
```bash
streamlit run src/compliance_assistant.py
```

## Project Structure

- `src/`: Source code directory
  - `compliance_assistant.py`: Main interactive assistant interface
  - `data_validator.py`: Data validation logic
  - `risk_scorer.py`: Risk scoring system
  - `anomaly_detector.py`: Anomaly detection module
  - `utils.py`: Utility functions
- `data/`: Directory for data files
- `tests/`: Test files
- `config/`: Configuration files

## Usage

1. Prepare your transaction data in CSV format with the following columns:
   - customerid
   - account_balance
   - transaction_amount
   - reported_amount
   - currency
   - country
   - transaction_date
   - risk_score

2. Launch the application using Streamlit
3. Upload your CSV file through the web interface
4. Interact with the assistant to analyze and validate your data

## Validation Rules

The system implements the following validation rules:
1. Transaction amount vs reported amount validation (1% deviation allowed for cross-currency)
2. Non-negative account balance check
3. Valid ISO 4217 currency code validation
4. Country jurisdiction validation
5. Transaction date validation
6. High-risk transaction flagging

## Contributing

Feel free to submit issues and enhancement requests!
