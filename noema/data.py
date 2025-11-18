import math
import os
import re
from nltk import pos_tag, word_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from colorama import Back, Fore, Style, init
init(autoreset=True)

from tqdm import tqdm
tqdm.pandas()

from loguru import logger
logger.add("noema.log", rotation="5000 KB")

from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (
    lower_text,
    normalize_whitespace,
    remove_eol_characters,
    remove_punct,
    remove_stopwords,
)
from nomic import embed
import spacy
from itertools import combinations
nlp = spacy.load("en_core_web_sm")


from noema.sheethelper import SheetHelper
from noema.automation.git_merge import *
from noema.params import *
from noema.utils import *

from gensim.models import Word2Vec
model = Word2Vec.load(f"{MODEL_PATH}/word2vec_model.model")

@time_it
def load_google_sheet():
    sh = SheetHelper(
        sheet_url="https://docs.google.com/spreadsheets/d/1c-811fFJYT9ulCneTZ7Z8b4CK4feEDRheR0Zea5--d0/edit#gid=0",
        sheet_id=0,
    )
    data = sh.gsheet_to_df()
    data.columns = [
        "submission_id",
        "respondent-id",
        "time",
        "rating",
        "free_text",
        "do_better",
        "pcn",
        "surgery",
        "campaing_id",
        "logic",
        "campaign_rating",
        "campaign_freetext",
    ]

    data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d %H:%M:%S")

    data.sort_values(by="time", inplace=True)
    return data


@time_it
def load_local_data():
    df = pd.read_csv(f"{DATA_PATH}/noema_data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=False)
    return df

@time_it
def clean_data(df):
    # Copy the DataFrame to avoid modifying the original data
    cleaned_df = df.copy()
    # Apply the conditions and update the DataFrame
    cleaned_df.loc[cleaned_df["review_word_count"] < 4, "review"] = np.nan
    cleaned_df.dropna(inplace=True)
    return cleaned_df

@time_it
def word_count(df):
    df["review_word_count"] = df["review"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else np.nan)
    return df

@time_it
def build_review_df(data):
    free_text = data[['time','free_text', 'pcn', 'surgery']].copy()
    do_better = data[['time','do_better', 'pcn', 'surgery']].copy()
    campaign = data[['time', 'campaign_freetext', 'pcn', 'surgery', 'campaing_id']].copy()

    free_text['origin'] = 'feedback'
    do_better['origin'] = 'do_better'

    free_text.columns = ['time', 'review', 'pcn', 'surgery', 'origin']
    do_better.columns = ['time', 'review', 'pcn', 'surgery', 'origin']
    campaign.columns = ['time', 'review', 'pcn', 'surgery', 'origin']

    free_text.dropna(inplace=True)
    do_better.dropna(inplace=True)
    campaign.dropna(inplace=True)

    all_reviews = pd.concat([free_text, do_better, campaign], axis=0, ignore_index=False)
    all_reviews = word_count(all_reviews)
    all_reviews = clean_data(all_reviews)

    return all_reviews


def text_preprocessing(text):
    preprocessor = Preprocessor()
    # preprocessor.pipe(lower_text)
    preprocessor.pipe(remove_eol_characters)
    # preprocessor.pipe(remove_stopwords, args={'lang': 'en'})
    # preprocessor.pipe(remove_punct)
    preprocessor.pipe(normalize_whitespace)

    text = preprocessor.run(text)
    return text

@time_it
def concat_save_final_df(processed_df, new_df):
    logger.info("💾 Concat Dataframes to data.parquet successfully")
    combined_data = pd.concat([processed_df, new_df], ignore_index=True)
    combined_data.sort_values(by="time", inplace=True, ascending=True)
    # combined_data.to_parquet(f"{DATA_PATH}/data.parquet", index=False)
    combined_data.to_csv(f"{DATA_PATH}/noema_data.csv", encoding="utf-8", index=False)
    return True

@time_it
def add_text_embeddings(df: pd.DataFrame, text_column: str, embedding_column: str = 'embedding') -> pd.DataFrame:
    """
    Adds text embeddings to a DataFrame using Nomic's embed API.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the column containing text to embed.
        embedding_column (str): Name of the new column to store embeddings (default: 'embedding').

    Returns:
        pd.DataFrame: DataFrame with an additional column containing embeddings.
    """
    texts = df[text_column].tolist()

    output = embed.text(
        texts=texts,
        model='nomic-embed-text-v1.5',
        task_type='search_document',
        inference_mode='local',
    )

    df[embedding_column] = output['embeddings']
    return df

english_stop_words = [
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an",
    "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "couldn",
    "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't",
    "doing", "don", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
    "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's",
    "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more",
    "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor",
    "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't",
    "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some",
    "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't",
    "we", "were", "weren", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't",
    "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
]

@time_it
def richness_score(df):
    """
    Given a dataframe with 'review' and 'embedding' columns,
    compute linguistic, semantic, and actionability sub-scores + overall richness score.
    """
    results = []

    for _, row in tqdm(df.iterrows(), desc="Calculating richness scores", unit="text", total=df.shape[0]):
        text = row['review']
        embed = row['embedding']

# ----- 🅾️ Please check logic and statistical validity -------------------------------🅾️
        # --- Basic cleaning ---
        clean = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = [w for w in word_tokenize(clean) if w not in english_stop_words and len(w) > 2]

        # --- 1. Linguistic Depth ---
        word_count = len(tokens)
        unique_words = len(set(tokens))
        lexical_diversity = unique_words / word_count if word_count else 0
        avg_word_len = np.mean([len(w) for w in tokens]) if tokens else 0
        linguistic_depth = (lexical_diversity + avg_word_len / 10) / 2

        # --- 2. Semantic_dispersion ---
        semantic_dispersion = semantic_dispersion_calculator(clean, model, remove_stopwords=True, use_lemmas=True)

        # --- 2. Semantic Density ---
        semantic_density = np.std(embed)  # 🅾️ this gives a more even distribution.

        # --- 3. Actionability / Specificity ---
        tags = pos_tag(tokens)
        verbs = sum(1 for _, t in tags if t.startswith('VB'))
        nouns = sum(1 for _, t in tags if t.startswith('NN'))
        suggestions = sum(1 for w in tokens if w in ['must', 'should', 'need', 'ought', 'has to', 'needs', 'recommend', 'suggest', 'advise', 'improve', 'get better', 'fix', 'change', 'start', 'stop', 'reduce', 'increase', 'continue', 'follow up', 'crucial', 'essential', 'vital', 'important', 'necessary', 'critical', 'priority', 'a must', 'check', 'test', 'examine', 'assess', 're-evaluate', 'look into', 'find a solution', 'wait and see', 'expect', 'demand', 'require', 'insist', 'hope', 'ask', 'listen', 'explain', 'clarify', 'communicate', 'call', 'spend more time'])
        orig_actionability = (verbs + nouns + suggestions) / (word_count + 1e-6)
        # Tunable parameters
        N_MID = 8  # Word count where scaling factor is 0.5
        K_STEEPNESS = 0.7
        # Calculate the Sigmoid Scaling Factor
        exponent = -K_STEEPNESS * (word_count - N_MID)
        scaling_factor_sigmoid = 1 / (1 + math.exp(exponent))

        # Apply the scaling factor
        actionability = orig_actionability * scaling_factor_sigmoid

        results.append([linguistic_depth, semantic_dispersion, semantic_density, actionability])

# ----- 🅾️ End -----------------------------------------------------------------🅾️

    # convert list of lists into separate columns
    scores = pd.DataFrame(results, columns=['ling_depth', 'sem_dispersion', 'semantic_density', 'actionability'])

    # attach new columns to the original dataframe
    df = pd.concat([df.reset_index(drop=True), scores], axis=1)

    # --- Normalise each metric to 0-1 scale ---
    scaler = MinMaxScaler()
    df[['ling_depth', 'sem_dispersion', 'semantic_density', 'actionability']] = scaler.fit_transform(
        df[['ling_depth', 'sem_dispersion', 'semantic_density', 'actionability']]
    )

    # --- Combine into overall richness score ---
    df['richness_score'] = (
        0.2 * df['ling_depth'] +
        0.1 * df['sem_dispersion'] +
        0.3 * df['semantic_density'] +
        0.4 * df['actionability']
    )

    return df

# ----- 🅾️ Please check logic and statistical validity -------------------------------🅾️

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words      # optional: custom set here

def semantic_dispersion_calculator(review, model,
                                   remove_stopwords=True,
                                   use_lemmas=True):
    doc = nlp(str(review))                     # let spaCy handle casing

    words = [token.lemma_.lower() if use_lemmas else token.text.lower()
             for token in doc
             if token.is_alpha and
             (not remove_stopwords or not token.is_stop)]

    # Keep only words known to the model
    valid_words = [w for w in words if w in model.wv]
    if len(valid_words) < 2:
        return np.nan

    # Vectorise and compute cosine distance matrix
    vectors = np.vstack([model.wv[w] for w in valid_words])
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1e-12

    cosine_matrix = (vectors @ vectors.T) / np.outer(norms, norms)
    cosine_matrix = np.clip(cosine_matrix, -1, 1)

    triu_idx = np.triu_indices(len(valid_words), k=1)
    distances = 1 - cosine_matrix[triu_idx]
    return distances.mean()

# ----- 🅾️ End -----------------------------------------------------------------🅾️


if __name__ == "__main__":
    logger.info("🅾️ Noema - Data Pipeline")

    # Load new data from Google Sheet
    raw_data = load_google_sheet()
    logger.info("🧩 Google Sheet data loaded")

    # Load local data.csv to dataframe
    processed_data = load_local_data()
    logger.info("💾 noemo_data.csv Loadded")

    # Return new data for processing
    data = raw_data[~raw_data.index.isin(processed_data.index)]
    logger.info(f"🆕 New rows to process: {data.shape[0]}")

    if data.shape[0] == 0:
        logger.error("❌ Make Data terminated - No now rows")

    else:
        data = build_review_df(data)
        logger.info(f"🧹 Data cleaned - {data.shape[0]} rows")

        logger.info("📗 Text Preprocesssing with *NLPretext")
        data["review"] = data["review"].apply(
        lambda x: text_preprocessing(str(x)) if not pd.isna(x) else np.nan)

        data = add_text_embeddings(data, text_column='review', embedding_column='embedding')
        data = richness_score(data)
        data.dropna(subset=['richness_score'], inplace=True)
        concat_save_final_df(processed_data, data)
