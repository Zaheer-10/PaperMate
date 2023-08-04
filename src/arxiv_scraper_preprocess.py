import arxiv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from constants import PATH_DATA_BASE # constant is a string that represents the path to the data directory

query_keywords = [
    "\"image segmentation\"",
    "\"self-supervised learning\"",
    "\"representation learning\"",
    "\"image generation\"",
    "\"object detection\"",
    "\"transfer learning\"",
    "\"transformers\"",
    "\"adversarial training\"",
    "\"generative adversarial networks\"",
    "\"model compression\"",
    "\"few-shot learning\"",
    "\"natural language processing\"",
    "\"graph neural networks\"",
    "\"colorization\"",
    "\"depth estimation\"",
    "\"point cloud\"",
    "\"structured data\"",
    "\"optical flow\"",
    "\"reinforcement learning\"",
    "\"super resolution\"",
    "\"attention mechanisms\"",
    "\"tabular data\"",
    "\"unsupervised learning\"",
    "\"semi-supervised learning\"",
    "\"explainable AI\"",
    "\"radiance field\"",
    "\"decision tree\"",
    "\"time series analysis\"",
    "\"molecule generation\"",
    "\"large language models\"",
    "\"LLMs\"",
    "\"language models\"",
    "\"image classification\"",
    "\"document image classification\"",
    "\"encoder-decoder\"",
    "\"multimodal learning\"",
    "\"multimodal deep learning\"",
    "\"speech recognition\"",
    "\"generative models\"",
    "\"anomaly detection\"",
    "\"recommender systems\"",
    "\"robotics\"",
    "\"knowledge graphs\"",
    "\"cross-modal learning\"",
    "\"attention mechanisms\"",
    "\"unsupervised translation\"",
    "\"machine translation\"",
    "\"dialogue systems\"",
    "\"sentiment analysis\"",
    "\"question answering\"",
    "\"text summarization\"",
    "\"sequential modeling\"",
    "\"neurosymbolic AI\"",
    "\"fairness in AI\"",
    "\"transferable skills\"",
    "\"data augmentation\"",
    "\"neural architecture search\"",
    "\"active learning\"",
    "\"automated machine learning\"",
    "\"meta-learning\"",
    "\"domain adaptation\"",
    "\"time series forecasting\"",
    "\"weakly supervised learning\"",
    "\"self-supervised vision\"",
    "\"visual reasoning\"",
    "\"knowledge distillation\"",
    "\"hyperparameter optimization\"",
    "\"cross-validation\"",
    "\"explainable reinforcement learning\"",
    "\"meta-reinforcement learning\"",
    "\"generative models in NLP\"",
    "\"knowledge representation and reasoning\"",
    "\"zero-shot learning\"",
    "\"self-attention mechanisms\"",
    "\"ensemble learning\"",
    "\"online learning\"",
    "\"cognitive computing\"",
    "\"self-driving cars\"",
    "\"emerging AI trends\"",
    "\"Attention is all you need\"",
    "\"GPT\"",
    "\"BERT\"",
    "\"Transformers\"",
    "\"yolo\"",
    "\"speech recognition\"",
    "\"LSTM\"",
    "\"GRU\"",
    "\"BERT - Bidirectional Encoder Representation of Transformers\"",
    "\"Large Language Model\" ",
    "\"Stable diffusion\"",
    "\"Attention is all you need\"",
    "\"Encoder-Decoder\"",
     "\"Paper Recommendation systems\"",
     "\" Latent Dirichlet Allocation (LDA)\"",
     "\"Transformers\"",
     "\"Generative Pre-trained transform's\"",
]



def query_with_keywords(query , client) -> tuple:
    
    """
    Query the arXiv API for research papers based on a specific query and filter results by selected categories.
    
    Args:
        query (str): The search query to be used for fetching research papers from arXiv.
    
    Returns:
        tuple: A tuple containing three lists - terms, titles, and abstracts of the filtered research papers.
        
            terms (list): A list of lists, where each inner list contains the categories associated with a research paper.
            titles (list): A list of titles of the research papers.
            abstracts (list): A list of abstracts (summaries) of the research papers.
            urls (list): A list of URLs for the papers' detail page on the arXiv website.
    """
    
    # Create a search object with the query and sorting parameters.
    search = arxiv.Search(
        query=query,
        max_results=6000,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )
    
    # Initialize empty lists for terms, titles, abstracts, urls and ids.
    terms = []
    titles = []
    abstracts = []
    urls = []
    ids = []
    
    # For each result in the search...
    for res in tqdm(client.results(search), desc=query):
        # Check if the primary category of the result is in the specified list.
        if res.primary_category in ["cs.CV", "stat.ML", "cs.LG", "cs.AI" ,"cs.CL"]:
            # If it is, append the result's categories, title, summary, url and ids to their respective lists.
            terms.append(res.categories)
            titles.append(res.title)
            abstracts.append(res.summary)
            urls.append(res.entry_id)
            ids.append(res.entry_id.split('/')[-1])

    # Return the four lists.
    return terms, titles, abstracts, urls , ids

def main() -> None:

    client = arxiv.Client(num_retries=20, page_size=500)

    all_titles = []
    all_abstracts = []
    all_terms = []
    all_urls = []
    all_ids = []

    for query in query_keywords:
        terms, titles, abstracts, urls , ids = query_with_keywords(query)
        all_titles.extend(titles)
        all_abstracts.extend(abstracts)
        all_terms.extend(terms)
        all_urls.extend(urls)
        all_ids.extend(ids)
    print("\n→ Writing results to CSV file...\n")

    arxiv_data = pd.DataFrame({
    'titles': all_titles,
    'abstracts': all_abstracts,
    'terms': all_terms,
    'urls': all_urls,
    'ids':all_ids })

    
    arxiv_data.to_csv(PATH_DATA_BASE / "data.csv", index=False)

    print("\n→ Scraping completed!\n ")
    
    print("\n→ Dropping Duplicated 🐼\n ")
    
    arxiv_data = pd.read_csv(PATH_DATA_BASE / "data.csv")
    arxiv_data = arxiv_data[~arxiv_data['titles'].duplicated()] 
    
    arxiv_data.to_csv(PATH_DATA_BASE / 'Filtered_arxiv_papers.csv' ,index=False)
    print("\n→ Filtered data saved successfully ❤️‍🔥\n ")

if __name__ == '__main__':
    main()
