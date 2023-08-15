import re
import torch
import pickle
import torch
import base64
import PyPDF2
import requests
from io import BytesIO
from pathlib import Path
from .models import Paper
from datetime import datetime
from .models import RecentPaper 
from transformers import pipeline
from django.shortcuts import render
from transformers import pipeline
import xml.etree.ElementTree as ET
from django.shortcuts import render, get_object_or_404
from django.shortcuts import render, get_object_or_404
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Populating the 'Paper' database with data.
Paper.populate_database()


#model for text-summarization
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)


# ----------------------------------------------Index---------------------------------------------------------------------------------------
def index(request):
    
    """
    Render the index page of the Django web application.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content of the index page with recent research papers categorized for display.
    steps:
         1. Retrieving recent papers from different categories and passing through the context
    """
    
    ml_papers = RecentPaper.objects.filter(category='cs.CL').order_by("-published_date")[:3]
    nlp_papers = RecentPaper.objects.filter(category='cs.LG').order_by("-published_date")[:3]
    ai_papers = RecentPaper.objects.filter(category='cs.AI').order_by("-published_date")[:3]
    cv_papers = RecentPaper.objects.filter(category='cs.CV').order_by("-published_date")[:3]

    # print("ML Papers Count:", ml_papers.count())
    # print("NLP Papers Count:", nlp_papers.count())
    # print("AI Papers Count:", ai_papers.count())
    # print("CV Papers Count:", cv_papers.count())
    
    context = {
        'ml_papers': ml_papers,
        'nlp_papers': nlp_papers,
        'ai_papers': ai_papers,
        'cv_papers': cv_papers,
    }
    return render(request, 'index.html', context)


# ----------------------------------------------search_papers---------------------------------------------------------------------------------------

def search_papers(request):
    """
    Search for relevant research papers using a user query and recognized speech, and display recommended papers.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content showing recommended research papers based on the user's query.
    """
    # sourcery skip: assign-if-exp, boolean-if-exp-identity, extract-method, remove-unnecessary-cast, use-f string-for-concatenation
    
    # Setting up paths for pre-trained models and data.
    PATH_SENTENCES = Path.cwd() / "Models/Sentences"
    PATH_EMBEDDINGS = Path.cwd() / "Models/Embeddings"

    if request.method == 'POST':
        # Retrieving user input and recognized speech.
        query = request.POST.get('query', '')
        recognized_text = request.POST.get('recognized_text', '')

        # Combining query and recognized speech if available.
        if recognized_text:
            query += ' ' + recognized_text
        
        # Check if either the query or recognized_text is empty
        if not query.strip() and not recognized_text.strip():
            return render(request, 'index.html', {'error_message': 'Please enter a query or use speech input.'})
        
        # Load pre-trained SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_path = PATH_EMBEDDINGS / "Embeddings.pkl"
        sentences_path = PATH_SENTENCES / "Sentences.pkl"
 
        # Load pre-calculated sentence embeddings
        with open(sentences_path, 'rb') as f:
            sentences_data = pickle.load(f)
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)

        # Generate a prompt template based on the user query
        # prompt_template = f"Recommended ArXiv papers related to: '{query}'"
        prompt_template = f"Could you kindly generate top ArXiv paper recommendations based on : '{query}'? Your focus on recent research and relevant papers is greatly appreciated."

        # Encoding user query and calculating cosine similarity.
        query_embedding = model.encode([prompt_template])
        cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings_data)[0]

        # Get indices of top 4 similar papers
        top_indices = cosine_scores.argsort(descending=True)[:4]
        top_indices = top_indices.cpu().numpy()  # Convert to numpy array
        top_paper_titles = [sentences_data[i.item()] for i in top_indices]  # Access elements using integer indices

        # Get paper details from the database
        recommended_papers = Paper.objects.filter(title__in=top_paper_titles)
        
        if len(recommended_papers) == 0:
            search_error = True
        else:
            search_error = False
        return render(request, 'recommendations.html', {'papers': recommended_papers, 'recommended_papers': recommended_papers , 'search_error': search_error})
    return render(request, 'index.html')

# ----------------------------------------------Recommendations---------------------------------------------------------------------------------------

def recommendations(request):
    """
    Retrieve and render all research paper recommendations for display on the recommendations page.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content displaying all recommended research papers.
    """
    
    # Retrieving all papers from the 'Paper' model.
    papers = Paper.objects.all()
    return render(request, 'recommendations.html', {'papers': papers})

# ----------------------------------------------Q&A---------------------------------------------------------------------------------------

def qa_page(request):
    """
    Render the Q&A page of the Django web application.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content of the Q&A page.
    """
    return render(request, 'qa.html')
# ----------------------------------------------About---------------------------------------------------------------------------------------

def about(request):  
    """
    Render the 'About' page of the Django web application.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content of the 'About' page.
    """
    return render(request, 'about.html')

# -------------------------------------Summarization-------------------------------------------------------
def summarize_paper(request, paper_id):
    """
    Retrieve and render recommended research paper summaries for display on the summarization page along with the pdf viewer.

    Args:
        request (HttpRequest): The HTTP request made by the user(summarize).
        paper_id (int): Identifier for the specific paper being summarized (corresponding Paper ID).

    Returns:
        HttpResponse: The rendered HTML content displaying the summarized research paper and view the original pdf visually in the page.
    """

    
    paper = get_object_or_404(Paper, ids=paper_id)
    
    # Define the PDF URL
    pdf_url = f"https://arxiv.org/pdf/{paper.ids}.pdf"

    # Download the PDF content using requests
    response = requests.get(pdf_url)
    pdf_content = response.content
    
    pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")
    # Create a PDF reader
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))

    # Extract text from pages 2 to 5 (index 1 to 4)
    start_page = 1  # Page index to start from
    end_page = min(start_page + 4, len(pdf_reader.pages))  # End at page index 5 or last page, whichever comes first

    extracted_text = ""
    for page_number in range(start_page, end_page):
        page = pdf_reader.pages[page_number]
        extracted_text += page.extract_text()

    # Split the extracted text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    text_chunks = text_splitter.split_text(extracted_text)

    # Combine the chunks into final text
    final_text = "".join(text_chunks)
    final_text = re.sub(r'[^a-zA-Z0-9\s]', '', final_text)
    final_text   = re.sub(r'\S*@\S*\s?', '', final_text)
    final_text= final_text.rstrip()

    pipe_sum = pipeline('summarization', model = base_model,tokenizer = tokenizer,max_length = 512, min_length = 50)
    result = pipe_sum(final_text)
    result = result[0]['summary_text']
    print("pdf_base64:", pdf_base64[:100]) 
    
    context = {
        'paper': paper,
        'result' : result,
        'pdf_base64': pdf_base64,
        # 'pdf_text': pdf_text  # Pass the extracted PDF text to the template
    }
    
    return render(request, 'summarization.html', context)
    