from flask import Flask, request, jsonify
from flask_cors import CORS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import nltk
import re
import time
from nltk.corpus import stopwords

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

def model(user_ques_1, user_ques_2):
    # Stop word removal
    stop = stopwords.words('english')
    ques_1_nstop = ' '.join([word for word in user_ques_1.split() if word not in stop])
    ques_2_nstop = ' '.join([word for word in user_ques_2.split() if word not in stop])

    # Tokenize the questions
    tok_ques_1 = nltk.word_tokenize(ques_1_nstop)
    tok_ques_2 = nltk.word_tokenize(ques_2_nstop)
    tok_ques_1_str = ' '.join(map(str, tok_ques_1))
    tok_ques_2_str = ' '.join(map(str, tok_ques_2))

    # Lemmatize data
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

    ques_1_lemm_str = lemmatize_text(tok_ques_1_str)
    ques_2_lemm_str = lemmatize_text(tok_ques_2_str)

    # Convert lemmatized data to lower case
    ques_1_lemm_str = ques_1_lemm_str.lower()
    ques_2_lemm_str = ques_2_lemm_str.lower()

    # Text similarity scores obtained using 'paraphrase-MiniLM-L3-v2' BERT model
    st = time.time()
    embd1 = model1.encode(ques_1_lemm_str, convert_to_tensor=True)
    embd2 = model1.encode(ques_2_lemm_str, convert_to_tensor=True)
    cosine_scores1 = util.pytorch_cos_sim(embd1, embd2)
    et = time.time()
    elapsed_time = et - st

    # Text similarity scores obtained using BERT model 'all-distilroberta-v1'
    st = time.time()
    embd1 = model2.encode(ques_1_lemm_str, convert_to_tensor=True)
    embd2 = model2.encode(ques_2_lemm_str, convert_to_tensor=True)
    cosine_scores2 = util.pytorch_cos_sim(embd1, embd2)
    et = time.time()
    elapsed_time = et - st

    # Text similarity scores obtained using BERT model 'multi-qa-distilbert-cos-v1'
    st = time.time()
    embd1 = model3.encode(ques_1_lemm_str, convert_to_tensor=True)
    embd2 = model3.encode(ques_2_lemm_str, convert_to_tensor=True)
    cosine_scores3 = util.pytorch_cos_sim(embd1, embd2)
    et = time.time()
    elapsed_time = et - st

    # Calculate average similarity score
    avg_similarity_score = (cosine_scores1.item() + cosine_scores2.item() + cosine_scores3.item()) / 3

    return avg_similarity_score

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Define the endpoint for processing PDFs
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
                similarity = model(q1, q2)
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
    app.run(debug=True, port=5002, use_reloader=False)
