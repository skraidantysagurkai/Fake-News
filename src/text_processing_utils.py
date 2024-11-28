import re
import string
import unicodedata

import contractions
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# download nltk extensions
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# declare things :D that we will use
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# list of odd and unwanted symbols that will mess with the model, found using pyspellchecker lybrary
unwanted_symbols = ['–', '—', '‘', '“', '”', '•', '…', '☑', '➡', 'ツ', '¯', '°', '´', '¿', '\xad', '\u200e', '\u200a',
                    '\u200b', '\u200f']


# helper functions to process the text
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


# count stopwords and punctuations using spacy
def count_stop_punct(texts: list[str]) -> tuple[int, int]:
    docs = nlp.pipe(texts)
    stopword_counts = []
    punct_counts = []
    for doc in docs:
        stopword_counts.append(sum([token.is_stop for token in doc]))
        punct_counts.append(sum([token.is_punct for token in doc]))

    return stopword_counts, punct_counts


# remove stopwords and punctuations using spacy
def remove_stop_punct(texts):
    docs = nlp.pipe(texts)
    processed = []
    for doc in docs:
        processed.append(' '.join([token.text for token in doc if not token.is_stop and not token.is_punct]))

    return processed


def extract_entities(texts):
    docs = nlp.pipe(texts)
    people = []
    orgs = []
    for doc in docs:
        people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return people, orgs, len(people), len(orgs)


# add required pipe for operations
# will be used later to count polarities and subjectivity
def add_pipe(pipe):
    nlp.add_pipe(pipe)


# calculate polarity and subjectivity
def count_pol_sub(texts):
    pols = []
    subjects = []
    for text in texts:
        doc = nlp(text)
        pols.append(doc._.blob.polarity)
        subjects.append(doc._.blob.subjectivity)

    return pols, subjects


# calculate lexical richness
def lexical_richness(text: str) -> int:
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    unique_words = set(tokens)
    total_words = len(tokens)

    # Return lexical richness only if total words meet the threshold
    if total_words >= 10:
        return len(unique_words) / total_words
    else:
        return 0


def get_metrics(titles: list[str], texts: list[str]) -> pd.DataFrame:
    metrics_df = pd.Dataframe()
    # calculate metrics asynchonously
    metrics_df[['text_stopword_count', 'text_punct_count']] = count_stop_punct(texts)
    metrics_df[['title_stopword_count', 'title_punct_count']] = count_stop_punct(titles)
    metrics_df['processed_text'] = remove_stop_punct(texts)
    metrics_df[['txt_people_ents', 'txt_org_ents'
                                   'txt_ppl_ent_count', 'txt_org_ent_count']] = extract_entities(texts)

    # add pipe to the nlp model
    add_pipe('spacytextblob')
    metrics_df[['text_polarity', 'text_subjectivity']] = count_pol_sub(texts)
    metrics_df[['title_polarity', 'title_subjectivity']] = count_pol_sub(titles)
    metrics_df['text_lexical_richness'] = [lexical_richness(text) for text in texts]
    metrics_df['title_lexical_richness'] = [lexical_richness(title) for title in titles]

    return metrics_df
