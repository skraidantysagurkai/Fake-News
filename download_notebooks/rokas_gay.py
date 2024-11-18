import kagglehub
import polars as pl
import os
import contractions
from spellchecker import SpellChecker
import spacy
import re
import multiprocessing as mp
import unicodedata
import numpy as np
from spacytextblob.spacytextblob import SpacyTextBlob

def load() -> pl.DataFrame:
    # Download latest version
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

    print("Path to dataset files:", path)

    # Read the data
    fake_df = pl.read_csv(os.path.join(path, 'fake.csv'))
    true_df = pl.read_csv(os.path.join(path, 'true.csv'))

    # Add labels
    fake_df = fake_df.with_columns(pl.lit(0).alias("label"))
    true_df = true_df.with_columns(pl.lit(1).alias("label"))

    # Concatenate and shuffle the DataFrame
    joined_df = pl.concat([fake_df, true_df]).sample(fraction=1).with_row_index().select(pl.exclude("index"))

    return joined_df

unwanted_symbols = ['–', '—', '‘', '“', '”', '•', '…', '☑', '➡', 'ツ',  '¯','°', '´', '¿', '\xad', '\u200e', '\u200a', '\u200b', '\u200f']

def process_text(text: str) -> str:
    text = text.lower()
    # remove accent characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Turkey’s -> Turkey's
    text = re.sub(r'(?<=\w)’(?=s)', "'", text)
    # fix contractions: don't -> do not
    text = ' '.join(contractions.fix(word) for word in text.split())
    # remove urls
    text = re.sub(r"http[s]?://\S+|www\.\S+", '', text)
    # replace all whitespaces, including unicode spaces and tabs with regular space
    pattern = '[' + re.escape(''.join(unwanted_symbols)) + ']'  # Escapes special regex characters
    text = re.sub(pattern, '', text)    
    text = re.sub(r'\s', ' ', text)

    return text

def get_len_text(text: str) -> int:
    return len(text.split())

def get_hastag_count(text: str) -> int:
    return len(re.findall(r"#\w+", text))

def get_tag_count(text: str) -> int:
    return len(re.findall(r"@\w+", text))

def extract_urls(text):
    return len(re.findall(r"http[s]?://\S+|www\.\S+", text))

def process_chunks(chunk: pl.DataFrame) -> pl.DataFrame:
    chunk = chunk.with_columns(
        pl.col('text').map_elements(lambda x: get_len_text(x), return_dtype=pl.Int32).alias("text_word_count"),
        pl.col('text').map_elements(lambda x: get_hastag_count(x), return_dtype=pl.Int32).alias("text_hastag_count"),
        pl.col('text').map_elements(lambda x: get_tag_count(x), return_dtype=pl.Int32).alias("text_mention_count"),
        pl.col('text').map_elements(lambda x: extract_urls(x), return_dtype=pl.Int32).alias("text_url_count"),
        pl.col('text').map_elements(lambda x: process_text(x), return_dtype=pl.String).alias("processed_text"),
        pl.col('title').map_elements(lambda x: get_len_text(x), return_dtype=pl.Int32).alias("title_word_count"),
        pl.col('title').map_elements(lambda x: get_hastag_count(x), return_dtype=pl.Int32).alias("title_hastag_count"),
        pl.col('title').map_elements(lambda x: get_tag_count(x), return_dtype=pl.Int32).alias("title_mention_count"),
        pl.col('title').map_elements(lambda x: extract_urls(x), return_dtype=pl.Int32).alias("title_url_count")
    )

    return chunk
    
def start(df: pl.DataFrame) -> pl.DataFrame:
    num_partitions = 8
    chunk_size = len(df) // num_partitions
    chunks = []
    for i in range(num_partitions):
        start = i*chunk_size
        end = (i+1)*chunk_size
        chunks.append(df.slice(start, end))
    
    with mp.Pool(processes=num_partitions,) as pool:
        result_chunks = pool.map(process_chunks, chunks)
        
    final_df = pl.concat(result_chunks)

    return final_df

def spacy_proceses(df: pl.DataFrame) -> pl.DataFrame:
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")
    
    texts = list(df.get_column("text"))
    titles = list(df.get_column("title"))
    
    text_stopword_counts = []
    text_punct_counts = []
    text_sentiments = []
    text_subjectivities = []
    
    title_stopword_counts = []
    title_punct_counts = []
    title_sentiments = []
    title_subjectivities = []
    
    for doc in nlp.pipe(texts, n_process=4):
        stopword_count = sum(token.is_stop for token in doc)
        punct_count = sum(token.is_punct for token in doc)
        
        text_punct_counts.append(punct_count)
        text_stopword_counts.append(stopword_count)
        text_sentiments.append(doc._.blob.polarity)
        text_subjectivities.append(doc._.blob.subjectivity)
        
    for doc in nlp.pipe(titles, n_process=4):
        stopword_count = sum(token.is_stop for token in doc)
        punct_count = sum(token.is_punct for token in doc)
        
        title_punct_counts.append(punct_count)
        title_stopword_counts.append(stopword_count)
        title_sentiments.append(doc._.blob.polarity)
        title_subjectivities.append(doc._.blob.subjectivity)

    
    
    df = df.with_columns(
        text_punct_count=text_punct_counts,
        text_stopword_count=text_stopword_counts,
        text_sentiment=text_sentiments,
        text_subjectivity=text_subjectivities,
        title_punct_count=title_punct_counts,
        title_stopword_count=title_stopword_counts,
        title_sentiment=title_sentiments,
        title_subjectivity=title_subjectivities
    )
    return df

if __name__ == "__main__":
    df = load()

    # apply  non spacy processes
    # df = start(df)
    df = spacy_proceses(df)
    
    df = df.with_columns(
        (pl.col('text_word_count')/pl.col('text_stopword_count')).alias('text_word_stopword_ratio'),
        (pl.col('title_word_count')/pl.col('title_stopword_count')).alias('title_word_stopword_ratio'),
    )
    
    df.write_csv("final.csv")
    
    