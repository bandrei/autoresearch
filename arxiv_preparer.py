import os
import arxiv
import pypdf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import time

# Categories relevant to finance
CATEGORIES = [
    "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", 
    "q-fin.PM", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR",
    "econ.GN", "econ.TH", "econ.EM"
]

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data_arxiv")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_arxiv_papers(query, max_results=100):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return list(client.results(search))

def extract_text_from_pdf(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def prepare_arxiv_data(num_papers=50, papers_per_shard=10):
    query = " OR ".join([f"cat:{cat}" for cat in CATEGORIES])
    print(f"Searching arXiv for: {query}")
    
    papers = fetch_arxiv_papers(query, max_results=num_papers)
    print(f"Found {len(papers)} papers.")
    
    all_texts = []
    temp_pdf_dir = os.path.join(CACHE_DIR, "temp_pdfs")
    os.makedirs(temp_pdf_dir, exist_ok=True)
    
    for i, paper in enumerate(papers):
        print(f"[{i+1}/{len(papers)}] Processing: {paper.title}")
        pdf_filename = f"{paper.get_short_id()}.pdf"
        pdf_path = os.path.join(temp_pdf_dir, pdf_filename)
        
        try:
            # Check if PDF already exists locally
            if os.path.exists(pdf_path):
                print(f"  Found local PDF for {paper.get_short_id()}. Extracting...")
            else:
                # Exponential backoff for downloads
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        paper.download_pdf(dirpath=temp_pdf_dir, filename=pdf_filename)
                        break 
                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 30 # Wait 30, 60 seconds
                            print(f"\nRate limited (429). Waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise e

            text = extract_text_from_pdf(pdf_path)
            if text and len(text) > 500: # Basic filter for empty/short content
                all_texts.append(text)
                # Keep PDFs for future use / re-processing
                print(f"  Successfully extracted {paper.get_short_id()}. PDF preserved in cache.")
            else:
                print(f"  Extraction failed or text too short for {paper.get_short_id()}")
        
        # Respect arXiv API rate limits - increased to be safe for PDFs
        # with random jitter to mimic human-like behavior
        import random
        time.sleep(10 + random.uniform(1, 5))

    # Save as parquet shards
    num_shards = (len(all_texts) + papers_per_shard - 1) // papers_per_shard
    for i in range(num_shards):
        shard_texts = all_texts[i*papers_per_shard : (i+1)*papers_per_shard]
        df = pd.DataFrame({"text": shard_texts})
        table = pa.Table.from_pandas(df)
        shard_filename = f"shard_{i:05d}.parquet"
        pq.write_table(table, os.path.join(DATA_DIR, shard_filename))
        print(f"Saved {shard_filename} with {len(shard_texts)} papers.")

    print(f"ArXiv data preparation complete. Shards saved in {DATA_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch and prepare arXiv finance data.")
    parser.add_argument("--num-papers", type=int, default=100, help="Total papers to fetch.")
    parser.add_argument("--papers-per-shard", type=int, default=20, help="Papers per parquet shard.")
    args = parser.parse_args()
    
    prepare_arxiv_data(num_papers=args.num_papers, papers_per_shard=args.papers_per_shard)
