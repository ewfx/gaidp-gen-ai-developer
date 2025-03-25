import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import openai
from src.data_validator import DataValidator
from src.risk_scorer import RiskScorer
from src.anomaly_detector import AnomalyDetector
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

class ComplianceAssistant:
    def __init__(self):
        self.validator = DataValidator
        self.risk_scorer = RiskScorer()
        self.anomaly_detector = AnomalyDetector()
        self.chat_history = []

    def process_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process the uploaded data and return comprehensive analysis."""
        # Convert transaction_date to datetime
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Initialize validator
        validator = self.validator(df)
        
        # Run validations
        validation_results = validator.validate_all()
        validation_summary = validator.get_validation_summary()
        
        # Train and run risk scoring
        self.risk_scorer.train(df)
        risk_results = self.risk_scorer.score_transactions(df)
        risk_insights = self.risk_scorer.get_risk_insights(df)
        
        # Train and run anomaly detection
        self.anomaly_detector.train(df)
        anomaly_results = self.anomaly_detector.detect_anomalies(df)
        anomaly_insights = self.anomaly_detector.get_anomaly_insights(df)
        
        return {
            "validation_results": validation_results,
            "validation_summary": validation_summary,
            "risk_insights": risk_insights,
            "anomaly_insights": anomaly_insights,
            "processed_data": risk_results.merge(
                anomaly_results[['is_anomaly', 'anomaly_score']],
                left_index=True,
                right_index=True
            )
        }

    def get_ai_response(self, user_query: str, context: Dict[str, Any]) -> str:
        """Get AI-generated response for user query."""
        # Convert DataFrame to dict and handle non-serializable objects
        serializable_context = {
            "validation_results": context["validation_results"],
            "validation_summary": context["validation_summary"],
            "risk_insights": {
                "risk_distribution": context["risk_insights"]["risk_distribution"],
                "average_risk_score": float(context["risk_insights"]["average_risk_score"]),
                "high_risk_factors": context["risk_insights"]["high_risk_factors"]
            },
            "anomaly_insights": {
                "total_anomalies": context["anomaly_insights"]["total_anomalies"],
                "anomaly_rate": float(context["anomaly_insights"]["anomaly_rate"]),
                "top_anomalies": [
                    {
                        "transaction_id": str(anomaly["transaction_id"]),
                        "customer_id": str(anomaly["customer_id"]),
                        "amount": float(anomaly["amount"]),
                        "reported_amount": float(anomaly["reported_amount"]),
                        "currency": str(anomaly["currency"]),
                        "country": str(anomaly["country"]),
                        "anomaly_score": float(anomaly["anomaly_score"]),
                        "reason": str(anomaly["reason"])
                    }
                    for anomaly in context["anomaly_insights"]["top_anomalies"]
                ],
                "anomaly_patterns": context["anomaly_insights"]["anomaly_patterns"]
            }
        }

        messages = [
            {"role": "system", "content": """You are a compliance assistant expert. 
            Analyze the provided context and answer user queries about transaction data, 
            compliance rules, and risk patterns. Provide clear, concise explanations 
            and specific recommendations when needed."""},
            {"role": "user", "content": f"""
            Context: {json.dumps(serializable_context)}
            
            User Query: {user_query}
            """}
        ]
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content

def main():
    st.title("Regulatory Compliance Assistant")
    st.write("""
    Welcome to the Regulatory Compliance Assistant! 
    Upload your transaction data and interact with the AI to analyze compliance rules,
    detect anomalies, and assess risks.
    """)
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ComplianceAssistant()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # File upload
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Process data if not already processed
            if st.session_state.analysis_results is None:
                with st.spinner("Processing data..."):
                    st.session_state.analysis_results = st.session_state.assistant.process_data(df)
                st.success("Data processed successfully!")
            
            # Display tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "Data Overview", 
                "Validation Results", 
                "Risk Analysis",
                "Anomaly Detection"
            ])
            
            with tab1:
                st.subheader("Data Overview")
                st.dataframe(df)
                st.write("### Summary Statistics")
                st.write(df.describe())
            
            with tab2:
                st.subheader("Validation Results")
                results = st.session_state.analysis_results['validation_results']
                summary = st.session_state.analysis_results['validation_summary']
                
                st.write("### Validation Summary")
                st.write(f"Total Transactions: {summary['total_transactions']}")
                st.write(f"Total Issues: {summary['total_issues']}")
                
                st.write("### Issues by Type")
                for issue_type, count in summary['issues_by_type'].items():
                    st.write(f"- {issue_type}: {count}")
                
                st.write("### Detailed Results")
                for result in results:
                    st.write(f"- **{result['type']}**: {result['details']}")
            
            with tab3:
                st.subheader("Risk Analysis")
                risk_insights = st.session_state.analysis_results['risk_insights']
                
                st.write("### Risk Distribution")
                st.write(risk_insights['risk_distribution'])
                
                st.write(f"Average Risk Score: {risk_insights['average_risk_score']:.2f}")
                
                st.write("### High Risk Factors")
                st.write(risk_insights['high_risk_factors'])
            
            with tab4:
                st.subheader("Anomaly Detection")
                anomaly_insights = st.session_state.analysis_results['anomaly_insights']
                
                st.write(f"Total Anomalies: {anomaly_insights['total_anomalies']}")
                st.write(f"Anomaly Rate: {anomaly_insights['anomaly_rate']:.2f}%")
                
                st.write("### Top Anomalies")
                for anomaly in anomaly_insights['top_anomalies']:
                    st.write(f"""
                    - Transaction {anomaly['transaction_id']}:
                      * Amount: {anomaly['amount']}
                      * Reason: {anomaly['reason']}
                      * Score: {anomaly['anomaly_score']:.2f}
                    """)
            
            # Chat interface
            st.write("---")
            st.subheader("Interactive Assistant")
            user_query = st.text_input("Ask me anything about the analysis:")
            
            if user_query:
                with st.spinner("Generating response..."):
                    ai_response = st.session_state.assistant.get_ai_response(
                        user_query,
                        st.session_state.analysis_results
                    )
                    st.session_state.chat_history.append({
                        "user": user_query,
                        "assistant": ai_response,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Display chat history
                for message in st.session_state.chat_history:
                    st.write(f"**You:** {message['user']}")
                    st.write(f"**Assistant:** {message['assistant']}")
                    st.write("---")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
