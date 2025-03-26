# Interactive Data Analysis Assistant

A powerful Streamlit application that helps you analyze datasets using AI-powered rules and enables interactive conversations about the results. The application uses Google's Gemini AI model to provide intelligent data analysis and insights.

## Features

- 📊 Support for multiple data formats (CSV, Excel, TXT, DOCX, PDF)
- 🤖 AI-powered data analysis using Google's Gemini model
- 💬 Interactive chat interface for follow-up questions
- 📝 Customizable analysis rules
- 🔄 Real-time progress tracking for large datasets
- 📱 Responsive and user-friendly interface

## Prerequisites

- Python 3.8 or higher
- Google API key for Gemini AI
- Internet connection for API access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to:
- Local URL: http://localhost:8501
- Network URL: http://<your-ip>:8501

## Usage Guide

1. **Model Selection**
   - Use the sidebar to select your preferred Gemini AI model
   - Default model: gemini-1.5-flash-001

2. **Data Input**
   - Choose between file upload or text input
   - Supported file formats:
     - CSV files
     - Excel files (xlsx, xls)
     - Text files (txt)
     - Word documents (docx)
     - PDF files

3. **Rules Input**
   - Define your analysis rules
   - Choose between file upload or text input
   - Supported formats: txt, docx, pdf

4. **Analysis**
   - Click "Analyze Data" to start the analysis
   - Monitor progress in real-time
   - View results in the main section

5. **Interactive Chat**
   - Ask follow-up questions about the results
   - Get detailed explanations and insights
   - View chat history

## Troubleshooting

1. **API Key Issues**
   - Ensure your Google API key is valid
   - Check the `.env` file format
   - Verify internet connectivity

2. **File Upload Problems**
   - Check file format compatibility
   - Ensure file size is within limits
   - Verify file encoding (UTF-8 recommended)

3. **Analysis Errors**
   - Check data format and structure
   - Verify rules syntax
   - Ensure sufficient API quota

## Performance Tips

- For large datasets, the application processes data in chunks
- Progress is displayed in real-time
- Results are cached for faster follow-up questions
- Use appropriate chunk sizes based on your data volume

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the repository or contact the maintainers. 