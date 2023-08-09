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

Paper.populate_database()


def index(request):
    return render(request, 'index.html')

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

