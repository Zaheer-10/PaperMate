from django.shortcuts import redirect, render , HttpResponse
from django.shortcuts import render
import arxiv
from django.shortcuts import render
from django.http import JsonResponse
from .models import Paper
from sentence_transformers import SentenceTransformer, util
import pickle
from pathlib import Path
from django.shortcuts import render
from arxiv import Client
from crossref.restful import Works
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from django.shortcuts import render
from .models import RecentPaper  # Import your RecentPaper model
import speech_recognition as sr


Paper.populate_database()



def index(request):
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


def recommendations(request):
    return render(request , "recommendations.html" )
    


def qa_page(request):
    return render(request, 'qa.html')



def search_papers(request):
    # sourcery skip: assign-if-exp, boolean-if-exp-identity, remove-unnecessary-cast
    PATH_SENTENCES = Path.cwd() / "Models/Sentences"
    PATH_EMBEDDINGS = Path.cwd() / "Models/Embeddings"

    if request.method == 'POST':
        query = request.POST.get('query', '')
        recognized_text = request.POST.get('recognized_text', '')

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
        prompt_template = f"Recommended ArXiv papers related to: '{query}'"
        # Generate a prompt template based on the user query
        # prompt_template = f"Could you kindly generate top ArXiv paper recommendations based on : '{query}'? Your focus on recent research and relevant papers is greatly appreciated."

        # Encode user query and calculate cosine similarity
        query_embedding = model.encode([prompt_template])
        cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings_data)[0]

        # Get indices of top 5 similar papers
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

def recommendations(request):
    papers = Paper.objects.all()
    return render(request, 'recommendations.html', {'papers': papers})

# def about(request):
#     return render(request , 'about.html')

def about(request):  
    return render(request, 'about.html')