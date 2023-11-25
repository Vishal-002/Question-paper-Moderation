from flask import Flask, request, jsonify
from flask_cors import CORS
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
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

        # Use regular expressions to split the text into individual questions with numbers
        arr1 = re.split(r'(\d+\.)', question_paper1_text)
        arr2 = re.split(r'(\d+\.)', question_paper2_text)

        # Combine question numbers with questions
        questions1 = [f"{i + 1}. {ques.strip()}" for i, ques in enumerate(arr1[2::2])]
        questions2 = [f"{i + 1}. {ques.strip()}" for i, ques in enumerate(arr2[2::2])]

        new_paper = []

        # Calculate similarity and create a new set of questions
        for q1 in questions1:
            for q2 in questions2:
                similarity = model(q1, q2)
                if similarity > 0.7:
                    new_paper.append(q1)
                    break

        # Add questions from question_paper2 that didn't have matches
        for q2 in questions2:
            if q2 not in new_paper:
                new_paper.append(q2)

        for q1 in questions1:
            if q1 not in new_paper:
                new_paper.append(q1)

        # Generate PDF
        output_file = 'output.pdf'
        doc = SimpleDocTemplate(output_file, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()
        style = styles['Normal']
        style.alignment = TA_CENTER

        # Add processed questions to the PDF
        for question in new_paper:
            paragraph = Paragraph(question, style)
            story.append(paragraph)
            story.append(Spacer(1, 5))

        # Build the PDF document
        doc.build(story)

        print(f'PDF file "{output_file}" generated successfully.')

        response = {
            'success': True,
            'message': 'PDFs processed successfully.',
            'pdf_file': output_file,
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
