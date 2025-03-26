import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
import io
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Load environment variables
load_dotenv()

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'current_rules' not in st.session_state:
    st.session_state.current_rules = None
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'result_history' not in st.session_state:
    st.session_state.result_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'gemini-1.5-flash-001'
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""
if 'should_clear_input' not in st.session_state:
    st.session_state.should_clear_input = False
if 'initial_results' not in st.session_state:
    st.session_state.initial_results = None
if 'current_rules_hash' not in st.session_state:
    st.session_state.current_rules_hash = None

def read_docx(file):
    """Read content from a Word document"""
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(file):
    """Read content from a PDF document"""
    pdf = PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf.pages])

def read_csv(file):
    """Read content from a CSV file"""
    content = file.read().decode('utf-8')
    return content

def read_txt(file):
    """Read content from a text file"""
    content = file.read().decode('utf-8')
    return content

def get_available_models():
    """Get list of available models"""
    try:
        available_models = genai.list_models()
        return {model.name: model.name for model in available_models}
    except Exception as e:
        st.error(f"Error listing models: {str(e)}")
        return {}

def initialize_model(model_name):
    """Initialize the selected model"""
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error initializing model {model_name}: {str(e)}")
        return None

@lru_cache(maxsize=100)
def get_model_response(prompt, model_name):
    """Cached function to get model response"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def process_chunk(chunk_data, rules, chunk_number, total_chunks, model_name):
    """Process a single chunk of data"""
    try:
        # Convert chunk to a list of dictionaries with proper string handling
        chunk_data = chunk_data.replace({pd.NA: None}).to_dict('records')
        # Convert any non-string values to strings
        for record in chunk_data:
            for key, value in record.items():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    record[key] = str(value)
        
        prompt = f"""
        You are a data analysis assistant. Analyze the following dataset chunk based on the given rules.
        This is chunk {chunk_number} of {total_chunks}.
        
        Dataset Chunk: {json.dumps(chunk_data, indent=2, ensure_ascii=False)}
        Rules: {rules}
        
        IMPORTANT: Your response must be a valid JSON object in exactly this format:
        {{
            "filtered_results": [
                // List of items that match the rules
            ],
            "explanations": [
                // List of explanations for each result
            ]
        }}
        
        Do not include any text before or after the JSON object.
        Do not use markdown formatting.
        Ensure all strings are properly escaped.
        Do not include any comments in the JSON.
        """
        
        response_text = get_model_response(prompt, model_name)
        
        # Clean and parse the response
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Remove any text before the first { and after the last }
        response_text = re.sub(r'^[^{]*({.*})[^}]*$', r'\1', response_text, flags=re.DOTALL)
        
        # Remove any trailing commas
        response_text = re.sub(r',\s*}', '}', response_text)
        response_text = re.sub(r',\s*]', ']', response_text)
        
        # Remove any comments
        response_text = re.sub(r'//.*?\n', '', response_text)
        
        # Remove any whitespace between elements
        response_text = re.sub(r'\s+', ' ', response_text)
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                # Fix unescaped quotes
                response_text = re.sub(r'(?<!\\)"', '\\"', response_text)
                # Fix missing quotes around keys
                response_text = re.sub(r'(\w+):', r'"\1":', response_text)
                # Fix missing quotes around string values
                response_text = re.sub(r':\s*([^"\'\d\[\]{}]+)', r':"\1"', response_text)
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If all parsing attempts fail, return empty results with error message
                return {
                    "filtered_results": [],
                    "explanations": [
                        f"Error parsing chunk {chunk_number} response: {str(e)}",
                        "Raw response: " + response_text[:200] + "..."  # Include first 200 chars of raw response
                    ]
                }
        
        # Validate result structure
        if not isinstance(result, dict):
            return {
                "filtered_results": [],
                "explanations": [f"Invalid response format in chunk {chunk_number}: Expected dictionary"]
            }
        
        if "filtered_results" not in result or "explanations" not in result:
            return {
                "filtered_results": [],
                "explanations": [f"Missing required fields in chunk {chunk_number} response"]
            }
        
        return result
    except Exception as e:
        return {
            "filtered_results": [],
            "explanations": [f"Error processing chunk {chunk_number}: {str(e)}"]
        }

def analyze_data(dataset, rules, follow_up_question=None, context_results=None):
    """Use Gemini AI to analyze data based on rules"""
    if not st.session_state.selected_model:
        st.error("Please select a model from the sidebar first")
        return None
        
    # Handle follow-up questions differently
    if follow_up_question:
        prompt = f"""
        Based on the following analysis results, please answer this follow-up question in a conversational format:
        
        Analysis Results: {json.dumps(context_results, indent=2, ensure_ascii=False)}
        Follow-up Question: {follow_up_question}
        
        Provide your response in a clear, conversational format. Focus on answering the specific question.
        Make sure to reference specific parts of the analysis results in your response.
        Do not include any JSON formatting or technical details.
        """
        
        try:
            response = get_model_response(prompt, st.session_state.selected_model)
            return response  # Return the raw response for chat
        except Exception as e:
            st.error(f"Error processing follow-up question: {str(e)}")
            return None
    
    # Handle initial data analysis
    if isinstance(dataset, pd.DataFrame):
        # Handle large datasets by chunking with parallel processing
        chunk_size = 1000  # Increased chunk size for faster processing
        total_records = len(dataset)
        total_chunks = (total_records + chunk_size - 1) // chunk_size
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        all_explanations = []
        error_chunks = []
        
        # Process chunks in parallel with increased workers
        with ThreadPoolExecutor(max_workers=5) as executor:  # Increased number of workers
            future_to_chunk = {}
            
            # Submit all chunks for processing
            for start_idx in range(0, total_records, chunk_size):
                end_idx = min(start_idx + chunk_size, total_records)
                chunk = dataset.iloc[start_idx:end_idx]
                chunk_number = start_idx // chunk_size + 1
                
                future = executor.submit(
                    process_chunk,
                    chunk,
                    rules,
                    chunk_number,
                    total_chunks,
                    st.session_state.selected_model
                )
                future_to_chunk[future] = chunk_number
            
            # Process completed chunks
            completed_chunks = 0
            for future in as_completed(future_to_chunk):
                chunk_number = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    if "filtered_results" in chunk_result:
                        all_results.extend(chunk_result["filtered_results"])
                    if "explanations" in chunk_result:
                        # Check if this is an error message
                        if any("Error parsing chunk" in exp or "Error processing chunk" in exp for exp in chunk_result["explanations"]):
                            error_chunks.append(chunk_number)
                        all_explanations.extend(chunk_result["explanations"])
                    
                    # Update progress
                    completed_chunks += 1
                    progress = completed_chunks / total_chunks
                    progress_bar.progress(progress)
                    status_text.text(f"Processed chunk {completed_chunks} of {total_chunks}")
                    
                except Exception as e:
                    error_chunks.append(chunk_number)
                    st.warning(f"Error processing chunk {chunk_number}: {str(e)}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Add summary of errors if any occurred
        if error_chunks:
            all_explanations.append(f"Note: Errors occurred in chunks {', '.join(map(str, error_chunks))}")
        
        # Combine all results
        result = {
            "filtered_results": all_results,
            "explanations": all_explanations
        }
        return result
    
    # Handle non-DataFrame data (text input)
    else:
        # Handle non-DataFrame data (text input)
        prompt = f"""
        You are a data analysis assistant. Analyze the following dataset based on the given rules.
        
        Dataset: {json.dumps(dataset, indent=2, ensure_ascii=False)}
        Rules: {rules}
        
        IMPORTANT: Your response must be a valid JSON object in exactly this format:
        {{
            "filtered_results": [
                // List of items that match the rules
            ],
            "explanations": [
                // List of explanations for each result
            ]
        }}
        
        Do not include any text before or after the JSON object.
        Do not use markdown formatting.
        Ensure all strings are properly escaped.
        """
        
        try:
            response_text = get_model_response(prompt, st.session_state.selected_model)
            # Clean and parse the response
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            response_text = re.sub(r'^[^{]*({.*})[^}]*$', r'\1', response_text, flags=re.DOTALL)
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                response_text = re.sub(r',\s*}', '}', response_text)
                response_text = re.sub(r',\s*]', ']', response_text)
                response_text = re.sub(r'//.*?\n', '', response_text)
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    result = {
                        "filtered_results": [],
                        "explanations": [
                            f"Error parsing response: {str(e)}",
                            f"Raw response: {response_text}"
                        ]
                    }
            
            return result
        except Exception as e:
            st.error(f"Error analyzing data: {str(e)}")
            return None

def display_results(results, is_follow_up=False):
    """Display results in a more visual format"""
    if not results:
        return
    
    if is_follow_up:
        st.markdown("### Latest Analysis")
    else:
        st.markdown("### Analysis Results")
    
    # Convert results to DataFrame if it's not already
    if isinstance(results, dict):
        if "filtered_results" in results:
            if results["filtered_results"]:
                # Convert filtered results to DataFrame
                df = pd.DataFrame(results["filtered_results"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No matching results found.")
            
            # Display explanations in a more readable format
            if "explanations" in results and results["explanations"]:
                st.markdown("#### Explanations")
                for i, explanation in enumerate(results["explanations"], 1):
                    st.markdown(f"**{i}.** {explanation}")
        else:
            st.write(results)
    else:
        st.write(results)

def main():
    # Set page config must be the first Streamlit command
    st.set_page_config(page_title="Interactive Data Analysis Assistant", layout="wide")
    
    # Initialize analysis state
    if 'is_analyzing' not in st.session_state:
        st.session_state.is_analyzing = False
    
    # Sidebar model selection
    st.sidebar.title("Model Settings")
    available_models = get_available_models()
    if available_models:
        # Find the default model in available models
        default_model = 'gemini-1.5-flash-001'
        model_keys = list(available_models.keys())
        
        # Try to find the exact model name first
        if default_model in model_keys:
            default_index = model_keys.index(default_model)
        else:
            # Try to find a model that contains the default model name
            matching_models = [i for i, key in enumerate(model_keys) if default_model in key]
            if matching_models:
                default_index = matching_models[0]
            else:
                default_index = 0
        
        selected_model = st.sidebar.selectbox(
            "Select AI Model",
            options=model_keys,
            format_func=lambda x: x.split('/')[-1],
            key="model_selector",
            index=default_index
        )
        st.session_state.selected_model = selected_model
        st.sidebar.info(f"Selected Model: {selected_model.split('/')[-1]}")
    else:
        st.sidebar.error("No models available. Please check your API key.")

    st.title("📊 Interactive Data Analysis Assistant")
    st.markdown("""
    This application helps you analyze datasets using AI-powered rules and allows you to have interactive conversations about the results.
    """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset")
        data_input_type = st.radio(
            "Choose input type",
            ["File Upload", "Text Input"]
        )
        
        if data_input_type == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload your data file",
                type=['csv', 'xlsx', 'xls', 'txt', 'docx', 'pdf']
            )
            if uploaded_file:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension in ['csv', 'xlsx', 'xls']:
                        if file_extension == 'csv':
                            dataset = pd.read_csv(uploaded_file)
                        else:
                            dataset = pd.read_excel(uploaded_file)
                        st.dataframe(dataset)
                        st.session_state.current_dataset = dataset
                    else:
                        if file_extension == 'docx':
                            content = read_docx(uploaded_file)
                        elif file_extension == 'pdf':
                            content = read_pdf(uploaded_file)
                        elif file_extension == 'txt':
                            content = read_txt(uploaded_file)
                        st.text_area("File Content", content, height=200)
                        st.session_state.current_dataset = content
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        else:
            dataset_input = st.text_area(
                "Enter your dataset (JSON format or plain text)",
                height=200
            )
            st.session_state.current_dataset = dataset_input
    
    with col2:
        st.subheader("Rules")
        rules_input_type = st.radio(
            "Choose rules input type",
            ["File Upload", "Text Input"]
        )
        
        if rules_input_type == "File Upload":
            uploaded_rules = st.file_uploader(
                "Upload rules document",
                type=['txt', 'docx', 'pdf']
            )
            if uploaded_rules:
                try:
                    file_extension = uploaded_rules.name.split('.')[-1].lower()
                    if file_extension == 'docx':
                        rules_input = read_docx(uploaded_rules)
                    elif file_extension == 'pdf':
                        rules_input = read_pdf(uploaded_rules)
                    else:  # txt
                        rules_input = read_txt(uploaded_rules)
                    st.text_area("Extracted Rules", rules_input, height=200)
                    # Check if rules have changed
                    new_rules_hash = hash(rules_input)
                    if st.session_state.current_rules_hash != new_rules_hash:
                        st.session_state.chat_history = []  # Clear chat history
                        st.session_state.current_rules_hash = new_rules_hash
                    st.session_state.current_rules = rules_input
                except Exception as e:
                    st.error(f"Error reading document: {str(e)}")
        else:
            rules_input = st.text_area(
                "Enter your rules",
                height=200
            )
            # Check if rules have changed
            new_rules_hash = hash(rules_input)
            if st.session_state.current_rules_hash != new_rules_hash:
                st.session_state.chat_history = []  # Clear chat history
                st.session_state.current_rules_hash = new_rules_hash
            st.session_state.current_rules = rules_input
    
    # Analyze button with state management
    if st.button("Analyze Data", disabled=st.session_state.is_analyzing):
        if not st.session_state.is_analyzing:
            st.session_state.is_analyzing = True
            try:
                if st.session_state.current_dataset is None or st.session_state.current_rules is None:
                    st.error("Please provide both dataset and rules first")
                    return
                
                results = analyze_data(st.session_state.current_dataset, st.session_state.current_rules)
                
                if results:
                    # Store initial results
                    st.session_state.initial_results = results
                    st.session_state.current_results = results
                    
                    # Add to result history
                    st.session_state.result_history.append({
                        "timestamp": pd.Timestamp.now(),
                        "results": results,
                        "type": "initial"
                    })
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Initial analysis complete. Found {len(results.get('filtered_results', []))} matching results."
                    })
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
            finally:
                st.session_state.is_analyzing = False
                st.rerun()
    
    # Display results (either initial or current)
    if st.session_state.current_results:
        display_results(st.session_state.current_results)
    
    # Chat interface for follow-up questions
    if st.session_state.current_results:
        st.markdown("---")
        st.subheader("Chat with AI about the Results")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"👤 You: {message['content']}")
            else:
                st.write(f"🤖 Assistant: {message['content']}")
        
        # Chat input with form to prevent rerun
        with st.form("chat_form"):
            # Clear input if needed
            if st.session_state.should_clear_input:
                st.session_state.chat_input = ""
                st.session_state.should_clear_input = False
            
            user_question = st.text_input("Ask a follow-up question about the results:", key="chat_input")
            submitted = st.form_submit_button("Send")
            
            if submitted and user_question:
                # Add user question to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                
                # Get AI response using current results as context
                response = analyze_data(
                    st.session_state.current_dataset,
                    st.session_state.current_rules,
                    follow_up_question=user_question,
                    context_results=st.session_state.current_results
                )
                
                if response:
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Set flag to clear input on next render
                    st.session_state.should_clear_input = True
                    st.rerun()
                else:
                    st.error("Failed to get a response. Please try again.")
    
    # Display result history in sidebar
    if st.session_state.result_history:
        st.sidebar.markdown("### Analysis History")
        for i, history_item in enumerate(reversed(st.session_state.result_history)):
            with st.sidebar.expander(f"Analysis {len(st.session_state.result_history) - i}"):
                st.markdown(f"**Time:** {history_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Type:** {'Initial Analysis' if history_item['type'] == 'initial' else 'Follow-up Analysis'}")
                if history_item['type'] == 'initial':
                    display_results(history_item['results'], is_follow_up=False)
                else:
                    st.write(history_item['results'])

if __name__ == "__main__":
    main() 