import multiprocessing as mp
import os
import re
import shutil

import kagglehub
import pandas as pd

from src.text_processing_utils import get_metrics, process_text


def load() -> tuple[pd.DataFrame, str]:
    # Download latest version
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    print(path)
    # Read the data
    fake_df = pd.read_csv(os.path.join(path, 'fake.csv'))
    true_df = pd.read_csv(os.path.join(path, 'true.csv'))

    # Add labels
    fake_df['label'] = 0
    true_df['label'] = 1

    # Concatenate and shuffle the DataFrame
    joined_df = pd.concat([fake_df, true_df]).sample(frac=1)

    return joined_df, path


# processing function to count metrics we are interested in
def process_chunks(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk['text_word_count'] = chunk['text'].apply(lambda text: len(text.split())).astype('int16')
    chunk['text_hastag_count'] = chunk['text'].apply(lambda text: len(re.findall(r"#\w+", text))).astype('int16')
    chunk['text_mention_count'] = chunk['text'].apply(lambda text: len(re.findall(r"@\w+", text))).astype('int16')
    chunk['text_url_count'] = chunk['text'].apply(
        lambda text: len(re.findall(r"http[s]?://\S+|www\.\S+", text))).astype('int16')
    chunk['processed_text'] = chunk['text'].apply(lambda x: process_text(x)).astype(str)
    chunk['title_word_count'] = chunk['title'].apply(lambda title: len(title.split())).astype('int16')
    chunk['title_hastag_count'] = chunk['title'].apply(lambda title: len(re.findall(r"#\w+", title))).astype('int16')
    chunk['title_mention_count'] = chunk['title'].apply(lambda title: len(re.findall(r"@\w+", title))).astype('int16')
    chunk['title_url_count'] = chunk['title'].apply(
        lambda title: len(re.findall(r"http[s]?://\S+|www\.\S+", title))).astype('int16')

    return chunk


# non spacy text processing using multiprocessing library
def process_1(df: pd.DataFrame) -> pd.DataFrame:
    num_partitions = 4
    chunk_size = len(df) // num_partitions
    chunks = []
    for i in range(num_partitions):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunks.append(df.iloc[start: end])
    with mp.Pool(processes=num_partitions, ) as pool:
        result_chunks = pool.map(process_chunks, chunks)
    final_df = pd.concat(result_chunks)

    return final_df


def process_2(df: pd.DataFrame):
    interval = df.shape[0] // 10
    dfs = []
    # split original dataframe into 10 parts
    for i in range(10):
        dfs.append(df.iloc[i * interval:(i + 1) * interval])
    processed_dfs = []

    # calculate metrics for each part
    for df in dfs:
        metrics_df = get_metrics(df['title'].tolist(), df['text'].tolist())
        df = pd.concat([df, metrics_df], axis=1)
        processed_dfs.append(df)

    # join the parts into final dataframe
    final_df = pd.concat(processed_dfs, axis=0)

    return final_df


if __name__ == "__main__":
    # load the dataframe
    df, path = load()
    # drop unwanted columns
    df.drop(["subject", "date"], axis=1, inplace=True)
    # process text without spacy
    df = process_1(df)
    df = process_2(df)
    # save dataframe
    df.to_csv("processed_data.csv")

    # remove downloaded dataframes\
    remove_path = os.path.abspath(os.path.join(path, "..", "..", ".."))
    shutil.rmtree(remove_path)
