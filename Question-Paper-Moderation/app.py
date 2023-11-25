from flask import Flask, request, jsonify
from flask_cors import CORS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import nltk
import re

app = Flask(__name__)
CORS(app)

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load BERT models
model1 = SentenceTransformer('paraphrase-MiniLM-L3-v2')
model2 = SentenceTransformer('all-distilroberta-v1')
model3 = SentenceTransformer('multi-qa-distilbert-cos-v1')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to process PDFs and return result
@app.route('/process_pdfs', methods=['POST'])
def process_pdfs():
    try:
        data = request.get_json()

        # Extract PDF paths from the request
        pdf_path1 = data.get('pdf_path1')
        pdf_path2 = data.get('pdf_path2')

        # Extract text from both PDF files
        question_paper1_text = extract_text_from_pdf(pdf_path1)
        question_paper2_text = extract_text_from_pdf(pdf_path2)

        # Use regular expressions to split the text into individual questions
        arr1 = re.split(r'\d+\.', question_paper1_text)
        arr2 = re.split(r'\d+\.', question_paper2_text)

        # Remove any leading or trailing whitespace from the questions
        arr1 = [question.strip() for question in arr1 if question.strip()]
        arr2 = [question.strip() for question in arr2 if question.strip()]

        new_paper = []

        for q1 in arr1:
            for q2 in arr2:
                similarity = util.pytorch_cos_sim(model1.encode(q1), model1.encode(q2))
                if similarity > 0.7:
                    new_paper.append(q1)
                    break

        # Add questions from question_paper2 that didn't have matches
        for q2 in arr2:
            if q2 not in new_paper:
                new_paper.append(q2)

        for q1 in arr1:
            if q1 not in new_paper:
                new_paper.append(q1)

        # Create a PDF document with A4 page size
        pdf_file = "output.pdf"
        c = canvas.Canvas(pdf_file, pagesize=A4)

        # Set font and other attributes
        c.setFont("Helvetica", 12)
        line_height = 14
        bottom_margin = 20

        # Calculate the available height on the page
        page_height = A4[1] - bottom_margin

        # Initialize page count
        page_number = 1

        # Iterate through the list and add the content to the PDF with pagination
        y_position = page_height

        for index, content in enumerate(new_paper, start=1):
            if y_position - line_height < 0:
                # If the content would overflow, start a new page
                c.showPage()
                c.setFont("Helvetica", 12)  # Reset font for the new page
                y_position = page_height
                page_number += 1

            content_with_index = f"{index}. {content}"
            c.drawString(20, y_position, content_with_index)
            y_position -= line_height

        # Save the PDF document
        c.save()

        response = {
            'success': True,
            'message': 'PDFs processed successfully.',
            'pdf_file': pdf_file,
        }
        return jsonify(response)

    except Exception as e:
        error_response = {
            'success': False,
            'message': f'Error processing PDFs: {str(e)}'
        }
        return jsonify(error_response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
