import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
import plotly.express as px
import spacy
from itertools import combinations
import math
# --- Setup ---
# Setup NLTK data (runs once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from gensim.models import Word2Vec
from noema.params import *
from noema.utils import *
from noema.ollama import ask_ollama

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words      # optional: custom set here
st.sidebar.header("Settings")

# Initialize the toggle state (optional, but good practice)
if 'model_toggle' not in st.session_state:
    st.session_state.model_toggle = False # Off by default
if 'debug_toggle' not in st.session_state:
    st.session_state.debug_toggle = False # Off by default
# Set the toggle value from session state
is_ollama_on = st.session_state.model_toggle
is_debug_on = st.session_state.debug_toggle
# Dynamically construct the label based on the current state
if is_ollama_on:
    # ON mode: Ollama is bold
    label_text = ":material/mode_off_on: :gray[OpenAI] | **Ollama**"
else:
    # OFF mode: OpenAI is bold
    label_text = ":material/mode_off_on: **OpenAI** | :gray[Ollama]"

if is_debug_on:
    # ON mode: Ollama is bold
    labeld_text = "**Debug ON**"
else:
    # OFF mode: OpenAI is bold
    labeld_text = ":gray[Debug OFF]"

debug = st.sidebar.toggle(labeld_text, value=is_debug_on, key='debug_toggle')
ollama = st.sidebar.toggle(
    label=label_text,
    value=is_ollama_on,
    key='model_toggle'
)
if is_ollama_on:
    model_choice = st.sidebar.selectbox("Select **Ollama** model:", ["qwen3:8b", "codellama:13b-instruct", "llama3.1:8b", "gpt-oss:20b"], index=2)
else:
    st.sidebar.write(":material/graph_7: Model **gpt-4o-mini**")
# --- Model Initialization ---

# Initialize models and API
# Use st.cache_resource to load models only once
@st.cache_resource
def get_models():
    """Loads and caches the ML models and stopwords."""
    stop_words = set(stopwords.words('english'))
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    model = Word2Vec.load(f"images/word2vec_model.model")
    nlp = spacy.load("en_core_web_sm")
    return stop_words, embedder, model, nlp

@st.cache_resource
def get_openai_client():
    """Initializes and caches the OpenAI client."""
    # Note: Streamlit handles secrets via st.secrets
    try:
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    except Exception:
        # Fallback if secrets aren't set, though API calls will fail
        client = OpenAI()
    return client

stop_words, embedder, model, nlp = get_models()
client = get_openai_client()


# --- Utility Functions ---
if ollama == False:
    def get_llm_richness(text):
        """Ask GPT model for richness score 1–5."""
        if not text or not isinstance(text, str):
            return np.nan

        prompt = f"""
        You are rating patient feedback for *richness and value*.
        Rate from 1–5:
        1 = vague or generic,
        3 = somewhat detailed,
        5 = highly rich, specific, and actionable.

        Feedback: "{text}"
        Rating (1-5):
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            match = re.findall(r"[1-5]", response.choices[0].message.content)
            return float(match[0]) if match else np.nan
        except Exception as e:
            st.warning(f"LLM error for text '{text[:50]}...': {str(e)}")
            return np.nan
elif ollama == True:
    def get_llm_richness(text):
        """Ask Ollama model for richness score 1–5."""
        if not text or not isinstance(text, str):
            return np.nan

        prompt = f"""
        You are rating patient feedback for *richness and value*.
        Rate from 1–5:
        1 = vague or generic,
        2 = somewhat specific - might mention a detail,
        3 = somewhat detailed - no actionable insights,
        4 = detailed and specific - at least one actionable insight,
        5 = highly rich, specific, and multiple actionable insights.
        Return only an integer rating from 1 to 5. No explanation is needed.
        Feedback: "{text}"
        """
        try:
            response = ask_ollama(
                user_prompt=prompt,
                system_prompt="You are a helpful AI assistant.",
                model=model_choice,
                format="plain text",
                temp=0,
                max_tokens=5,
            )
            match = re.findall(r"[1-5]", response)
            return float(match[0]) if match else np.nan
        except Exception as e:
            st.warning(f"LLM error for text '{text[:50]}...': {str(e)}")
            return np.nan

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



def compute_metrics(text):
    """Compute linguistic, semantic, and actionability metrics."""
    if not text or not isinstance(text, str):
        return 0, 0, 0

    clean = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = [w for w in word_tokenize(clean) if w not in stop_words and len(w) > 2]
    if not tokens:
        return 0, 0, 0

    # Linguistic depth
    word_count = len(tokens)
    unique_words = len(set(tokens))
    lexical_diversity = unique_words / word_count
    avg_word_len = np.mean([len(w) for w in tokens])
    ling_depth = (lexical_diversity + avg_word_len / 10) / 2

    # Semantic density
    semantic_dispersion = semantic_dispersion_calculator(clean, model, remove_stopwords=True, use_lemmas=True)

    # Actionability
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

    if debug == True:
        with st.expander(f":blue[**Debug Info**] - {text}", icon=":material/bug_report:"):
            st.success(f"Tokens: {tokens}")
            st.error(f"Word Count: {word_count}, Unique Words: {unique_words}, Ling Depth: {ling_depth:.3f} (Lexical Diversity: {lexical_diversity:.3f}, Avg Word Length: {avg_word_len:.2f})")
            st.error(f"Semantic Dispersion: {semantic_dispersion:.3f}")
            st.info(f"Verbs: {verbs}, Nouns: {nouns}, Suggestions: {suggestions}, Word Count: {word_count}")
            st.info(f"Original Actionability: {orig_actionability:.3f}")
            st.info(f"Scaling Factor (based on word count): {scaling_factor_sigmoid:.3f}")
            st.warning(f"Final Actionability: {actionability:.3f}")

    return ling_depth, semantic_dispersion, actionability

def richness_score_hybrid(df):
    """Compute hybrid richness scores for dataframe."""

    # 1. Compute Raw Metrics
    # We store them in '_raw' columns first
    metrics = df['text'].apply(lambda x: compute_metrics(str(x)))
    df[['ling_depth_raw', 'semantic_dispersion_raw', 'actionability_raw']] = pd.DataFrame(metrics.tolist(), index=df.index)

    # 2. Scale Metrics
    # Create new columns for scaled values, defaulting to raw values
    df['ling_depth'] = df['ling_depth_raw']
    df['semantic_dispersion'] = df['semantic_dispersion_raw']
    df['actionability'] = df['actionability_raw']

    # *** THIS IS THE FIX ***
    # Only apply scaling if there is more than one row.
    # MinMaxScaler will scale a single row to all zeros.
    if len(df) > 1:
        scaler = MinMaxScaler()
        cols_to_scale = ['ling_depth', 'semantic_dispersion', 'actionability']
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    elif len(df) == 1:
        # If only one row, scaling is undefined.
        # Set to a neutral 0.5 for the fusion formula.
        df[['ling_depth', 'semantic_dispersion', 'actionability']] = 0.5

    # 3. LLM Scoring
    # Use fillna(3) as a neutral score if API fails
    # ---- 1️⃣ Create a progress bar widget -----------------------------
    progress_placeholder = st.empty()
    progress_bar        = progress_placeholder.progress(0, text="Rating feedback…")

    # ---- 2️⃣ Score each line manually ----------------------------------
    scores = []
    total = len(df)

    for i, txt in enumerate(df['text'], start=1):
        scores.append(get_llm_richness(txt))          # single‑string call
        progress_bar.progress(i / total)

    # ---- 3️⃣ Clean up the bar and attach the result ------------------
    progress_placeholder.empty()
    df['llm_score'] = pd.Series(scores, index=df.index).fillna(3)

    # 4. Fusion
    # Use the *scaled* columns for fusion
    df['richness_score'] = (
        0.15 * df['ling_depth'] +
        0.25 * df['semantic_dispersion'] +
        0.35 * df['actionability'] +
        0.25 * (df['llm_score'] / 5)  # llm_score is already 1-5, so just normalize
    ) * 100

    return df

# --- Streamlit UI ---
st.set_page_config(page_title="Noema Richness Scorer", layout="centered")

st.image("images/richness_s.gif")
st.logo("images/logo.png", size='large')


st.write("""
This app evaluates the **richness and value** of free-text feedback using:
- **Linguistic depth** (complexity & vocabulary diversity)
- **Semantic density** (information variety)
- **Actionability** (presence of useful suggestions)
- **LLM-based contextual scoring**
""")

st.info("Note: LLM scoring requires an `OPENAI_API_KEY` to be set in your Streamlit secrets.", icon="🔑")


# Input Options
mode = st.radio("Input Mode", ["Enter Feedback", "Upload CSV"])
df_result = None

if mode == "Enter Feedback":
    text_input = st.text_area("Enter one or more feedback items (each line separate):",
                              height=200,
                              placeholder="e.g., The nurse was very kind.\nThe food should be warmer.\nI recommend improving the checkout process.")
    if text_input:
        feedbacks = [t.strip() for t in text_input.split("\n") if t.strip()]
        if feedbacks:
            df = pd.DataFrame({'text': feedbacks})
            if st.button("Compute Richness Scores"):
                with st.spinner("Scoring feedback... (LLM step may take a moment)"):
                    df_result = richness_score_hybrid(df.copy())
                st.success("✅ Scoring complete!")
        else:
            st.info("Please enter at least one line of feedback.")

elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV with a 'text' column", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded, encoding='latin1')
        if 'text' not in df.columns:
            st.error("CSV must have a column named 'text'.")
        else:
            df = df[['text']].dropna().drop_duplicates()
            df = df[6200:6400]
            if st.button(f"Compute Richness Scores for {len(df)} items"):
                with st.spinner(f"Scoring {len(df)} items... (LLM step may take a while)"):
                    df_result = richness_score_hybrid(df.copy())
                st.success("✅ Scoring complete!")

# --- Display Results ---
if df_result is not None:
    st.subheader("Scoring Results")

    # Define columns to show. We show the raw scores as they are more interpretable
    cols_to_show = [
        'text',
        'richness_score',
        'llm_score',
        'ling_depth_raw',
        'semantic_dispersion_raw',
        'actionability_raw'
    ]

    # Format the dataframe for better display
    df_display = df_result[cols_to_show].sort_values('richness_score', ascending=False)

    st.dataframe(
        df_display.style.format({
            'richness_score': '{:.1f}',
            'llm_score': '{:.0f}',
            'ling_depth_raw': '{:.3f}',
            'semantic_dispersion_raw': '{:.3f}',
            'actionability_raw': '{:.3f}',
        }),
        use_container_width=True
    )

    # Visualisation
    st.subheader("Results Visualization")
    if len(df_result) > 1:
        # Show histogram for multiple items
        # Distribution of Richness Scores
        fig_richness = px.histogram(
            df_result,
            x='richness_score',
            #nbins=20,
            # FIX: Use color_discrete_sequence for a uniform color
            color_discrete_sequence=['#96529c'],
            title="Distribution of Richness Scores",
            height=300
        )
        st.plotly_chart(fig_richness, use_container_width=True)

        # Distribution of actionability_raw
        fig_actionability = px.histogram(
            df_result,
            x='actionability_raw',
            #nbins=20,
            # FIX: Use color_discrete_sequence for a uniform color
            color_discrete_sequence=['#5a8ab4'],
            title="Distribution of actionability_raw",
            height=300
        )
        st.plotly_chart(fig_actionability, use_container_width=True)

        # Distribution of semantic_dispersion_raw
        fig_dispersion = px.histogram(
            df_result,
            x='semantic_dispersion_raw',
           # nbins=20,
            # FIX: Use color_discrete_sequence for a uniform color
            color_discrete_sequence=['#95c8e7'],
            title="Distribution of semantic_dispersion_raw",
            height=300
        )
        st.plotly_chart(fig_dispersion, use_container_width=True)

        # Distribution of ling_depth_raw
        fig_depth = px.histogram(
            df_result,
            x='ling_depth_raw',
            #nbins=20,
            # FIX: Use color_discrete_sequence for a uniform color
            color_discrete_sequence=['#ca6226'],
            title="Distribution of ling_depth_raw",
            height=300
        )
        st.plotly_chart(fig_depth, use_container_width=True)

    # Always show bar chart, but horizontal is better
    bar_title = "Richness Scores"
    if len(df_result) > 1:
        bar_title = "Top 20 Richest Feedback Items"
        df_display = df_display.head(20)

    # Truncate text for better plot display
    df_display['text_short'] = df_display['text'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

    fig_bar = px.bar(df_display.sort_values('richness_score', ascending=True),
                    x='richness_score', y='text_short',
                    orientation='h', title=bar_title,
                    hover_data=['text'])

    # Add the height parameter here (e.g., height=800)
    fig_bar.update_layout(yaxis_title="Feedback (truncated)", height=600)

    st.plotly_chart(fig_bar, use_container_width=True)
