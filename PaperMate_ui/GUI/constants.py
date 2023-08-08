from sentence_transformers import SentenceTransformer, util
import pickle
from pathlib import Path
from .models import Paper




PATH_SENTENCES = Path.cwd() / "Models/Sentences"
PATH_EMBEDDINGS = Path.cwd() / "Models/Embeddings"

query = 'Large Language Models'
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
top_indices = cosine_scores.argsort(descending=True)[:5]
top_paper_titles = [sentences_data[i] for i in top_indices]

# Print for debugging
print("Top paper titles:", top_paper_titles)

# Get paper details from the database
recommended_papers = Paper.objects.filter(title__in=top_paper_titles)

# Print for debugging
print("Recommended papers queryset:", recommended_papers)

