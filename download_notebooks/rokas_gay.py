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
    text = re.sub(r'https?:\S*', '', text)
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

def process_chunks(chunk: pl.DataFrame) -> pl.DataFrame:
    chunk = chunk.with_columns(
        pl.col('text').map_elements(lambda x: get_len_text(x), return_dtype=pl.Int32).alias("word_count"),
        pl.col('text').map_elements(lambda x: get_hastag_count(x), return_dtype=pl.Int32).alias("hastag_count"),
        pl.col('text').map_elements(lambda x: get_tag_count(x), return_dtype=pl.Int32).alias("tag_count"),
        pl.col('text').map_elements(lambda x: process_text(x), return_dtype=pl.String).alias("processed_text")
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
    
    stop_word_counts = []
    punct_counts = []
    cleared_texts = []
    
    for doc in nlp.pipe(df.get_column("text"), n_process=6):
        stopwords = 0
        puncts = 0
        for token in doc:
            if token.is_punct:
                puncts += 1
            if token.is_stop:
                stopwords += 1
            
        stop_word_counts.append(stopwords)
        punct_counts.append(puncts)
        
    for doc in nlp.pip(df.get_column("processed_text"), n_process=6):
        words = []
        for token in doc:
            if not token.is_stop and not token.is_punkt:
                words.append(token.text)
        cleared_texts.append(' '.join(words))
        
    df.with_columns(
        stop_count=stop_word_counts,
        punct_count=punct_counts,
        processed_text=cleared_texts
    )
    

if __name__ == "__main__":
    df = load()

    # apply  non spacy processes
    # df = start(df)
    df = spacy_proceses(df)
    print(df)
    
    