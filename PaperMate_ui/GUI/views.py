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


Paper.populate_database()


# def index(request):
#     return render(request, 'index.html')

def index(request):
    # Define the URL for the arXiv API query
    url = "http://export.arxiv.org/api/query?search_query=cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.CV&sortBy=lastUpdatedDate&sortOrder=descending&max_results=100"
    # Send a GET request to the URL and get the response
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the XML data from the response content
        root = ET.fromstring(response.content)

        # Initialize empty lists for each category
        ml_papers = []  # Machine learning papers
        ai_papers = []  # Artificial intelligence papers
        nlp_papers = []  # Natural language processing papers
        cv_papers = []  # Computer vision papers
        # Loop through each entry
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            # Extract the date string
            published_date_str = entry.find("{http://www.w3.org/2005/Atom}published").text
    
            # Convert the date string to a datetime object
            published_date = datetime.strptime(published_date_str, "%Y-%m-%dT%H:%M:%SZ")
    
            # Format the datetime object as a readable date string
            formatted_date = published_date.strftime("%d - %B - %Y")  # Example format: "09 - August - 2023"
            # Get the category of the paper
            category = entry.find("{http://www.w3.org/2005/Atom}category").attrib["term"]

            # Get paper details
            paper = {
                "link": entry.find("{http://www.w3.org/2005/Atom}id").text,
                "title": entry.find("{http://www.w3.org/2005/Atom}title").text,
                "authors": ", ".join([author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]),
                "summary": entry.find("{http://www.w3.org/2005/Atom}summary").text,
                "date": entry.find("{http://www.w3.org/2005/Atom}published").text,
            }

            # Append the entry to the corresponding list based on the category
            if category == "cs.LG" and len(ml_papers) < 3:
                ml_papers.append(paper)
            elif category == "cs.AI" and len(ai_papers) < 3:
                ai_papers.append(paper)
            elif category == "cs.CL" and len(nlp_papers) < 3:
                nlp_papers.append(paper)
            elif category == "cs.CV" and len(cv_papers) < 3:
                cv_papers.append(paper)

        # Create a context dictionary to pass the papers to the template
        context = {
            'ml_papers': ml_papers,
            'ai_papers': ai_papers,
            'nlp_papers': nlp_papers,
            'cv_papers': cv_papers,
            'formatted_date': formatted_date, 
        }

        # Render the template with the context
        return render(request, 'index.html', context)


def recommendations(request):
    return render(request , "recommendations.html" )
    


def qa_page(request):
    return render(request, 'qa.html')



def search_papers(request):
    PATH_SENTENCES = Path.cwd() / "Models/Sentences"
    PATH_EMBEDDINGS = Path.cwd() / "Models/Embeddings"

    if request.method == 'POST':
        query = request.POST.get('query', '')

        # Load pre-trained SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_path = PATH_EMBEDDINGS / "Embeddings.pkl"
        sentences_path = PATH_SENTENCES / "Sentences.pkl"

        # Load pre-calculated sentence embeddings
        with open(sentences_path, 'rb') as f:
            sentences_data = pickle.load(f)
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)

        # Encode user query and calculate cosine similarity
        query_embedding = model.encode([query])
        cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings_data)[0]
        
        # Get indices of top 5 similar papers
        top_indices = cosine_scores.argsort(descending=True)[:4]
        top_indices = top_indices.cpu().numpy()  # Convert to numpy array
        top_paper_titles = [sentences_data[i.item()] for i in top_indices]  # Access elements using integer indices

        # Get paper details from the database
        recommended_papers = Paper.objects.filter(title__in=top_paper_titles)
        # Print for debugging
        # print("Recommended papers queryset:", recommended_papers)


        return render(request, 'recommendations.html', {'papers': recommended_papers, 'recommended_papers': recommended_papers})


def recommendations(request):
    papers = Paper.objects.all()
    return render(request, 'recommendations.html', {'papers': papers})

