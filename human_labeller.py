"""
Human Labeller - Streamlit App
-------------------------------
A Streamlit app for human classification of GP reviews into topic categories.
Allows human labelers to classify reviews and save them to Google Sheets.
Includes aspect-based segmentation for multi-aspect reviews.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import spacy
from sentence_transformers import SentenceTransformer
import re

# Import local modules
from noema.params import DATA_PATH
from noema.sheethelper import SheetHelper
from noema.ollama import ask_ollama

# Topic categories from setfit_classification.py
TOPIC_LABELS = {
    0: "Appointment Availability & Access",
    1: "Telephone & Digital Access & Services",
    2: "Administrative Process",
    3: "Facilities & Cleanliness",
    4: "Prescription & Medication Management",
    5: "Clinical Treatment Quality",
    6: "Test Results & Follow-up",
    7: "Health Promotion",
    8: "Nursing & HCA Services",
    9: "Reception Staff Profesionalism & Attitude",
    10: "Clinical Staff Profesionalism & Attitude",
    11: "Continuity of Care",
    12: "Privacy & Confidentiality",
    13: "Feedback & Complaints Process",
    14: "Overall Experience",
}

# Google Sheet URL
SHEET_URL = "https://docs.google.com/spreadsheets/d/1NAoZrKt4itQgUt2K58DKH5my4ZS2SrHlWpVcHVui108/edit?gid=0#gid=0"

logger.add("human_labeller.log", rotation="1000 KB")


@st.cache_resource
def load_segmentation_models():
    """
    Load and cache models for segmentation.

    Returns:
        tuple: (SentenceTransformer model, spacy model)
    """
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        nlp = spacy.load("en_core_web_sm")
        logger.info("Segmentation models loaded successfully")
        return embedder, nlp
    except Exception as e:
        logger.error(f"Error loading segmentation models: {e}")
        st.error(f"Error loading segmentation models: {e}")
        return None, None


def segment_review_sentence_transformer(text, embedder, threshold=0.3):
    """
    Segment review using SentenceTransformer embeddings to detect topic shifts.

    Args:
        text (str): Review text
        embedder: SentenceTransformer model
        threshold (float): Similarity threshold for topic shift detection

    Returns:
        list[str]: List of text segments
    """
    if not text or not isinstance(text, str):
        return [text]

    try:
        # Split into clauses (sentences and comma-separated parts)
        clauses = re.split(r'[.,;]', text)
        clauses = [c.strip() for c in clauses if c.strip()]

        if len(clauses) <= 1:
            return [text]

        # Embed each clause
        embeddings = embedder.encode(clauses)

        # Calculate cosine similarities between consecutive clauses
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(sim)

        # Detect topic shifts and create segments
        segments = []
        current_segment = clauses[0]

        for i, sim in enumerate(similarities):
            if sim < threshold:  # Low similarity = new topic
                segments.append(current_segment.strip())
                current_segment = clauses[i+1]
            else:
                current_segment += ". " + clauses[i+1]

        # Add the last segment
        segments.append(current_segment.strip())

        # Filter out very short segments (< 3 words)
        segments = [s for s in segments if len(s.split()) >= 3]

        if not segments:
            return [text]

        logger.info(f"Segmented review into {len(segments)} parts using SentenceTransformer")
        return segments

    except Exception as e:
        logger.error(f"Error in SentenceTransformer segmentation: {e}")
        return [text]


def segment_review_spacy(text, nlp):
    """
    Segment review using spaCy NLP to detect aspect boundaries.

    Args:
        text (str): Review text
        nlp: spaCy model

    Returns:
        list[str]: List of text segments
    """
    if not text or not isinstance(text, str):
        return [text]

    try:
        doc = nlp(text)
        segments = []

        for sent in doc.sents:
            # Split into sub-clauses on commas for finer segmentation
            sub_clauses = [clause.strip() for clause in sent.text.split(',') if clause.strip()]

            for clause in sub_clauses:
                # Detect contrast or shift markers
                if any(word in clause.lower().split() for word in
                       ["but", "however", "although", "though", "yet", "while", "whereas"]):
                    segments.append(clause.strip())
                else:
                    segments.append(clause.strip())

        # Merge very short fragments (< 3 words)
        clean_segments = []
        buffer = ""
        for seg in segments:
            if len(seg.split()) < 3:
                buffer += " " + seg
            else:
                if buffer:
                    clean_segments.append((buffer + " " + seg).strip())
                    buffer = ""
                else:
                    clean_segments.append(seg.strip())

        if buffer:
            clean_segments.append(buffer.strip())

        if not clean_segments:
            return [text]

        logger.info(f"Segmented review into {len(clean_segments)} parts using spaCy")
        return clean_segments

    except Exception as e:
        logger.error(f"Error in spaCy segmentation: {e}")
        return [text]


def segment_review_llm(text, model="gpt-oss:20b"):
    """
    Segment review using LLM (Ollama) for aspect-based segmentation.

    Args:
        text (str): Review text
        model (str): Ollama model name

    Returns:
        list[str]: List of text segments
    """
    if not text or not isinstance(text, str):
        return [text]

    try:
        prompt = f"""You are an expert in aspect-based sentiment analysis.

Task:
Perform aspect-based text segmentation on the following review.
Split the review into coherent, meaningful units that each discuss a single aspect or topic.
These segments will later be used for separate sentiment analysis and classification.

Instructions:
- Keep the text segments in their original order.
- Do not summarize or alter wording.
- Return the result as a Python list of strings.
- Each segment should focus on ONE specific aspect or topic.

Input review:
{text}

Output format (return ONLY the list, no other text):
["segment_1", "segment_2", "segment_3", ...]
"""

        response = ask_ollama(
            user_prompt=prompt,
            system_prompt="You are an expert in aspect-based text analysis. Return only the requested format.",
            model=model,
            format="Python list of strings",
            temp=0,
            max_tokens=1000
        )

        if response:
            # Try to parse the response as a Python list
            import ast
            try:
                segments = ast.literal_eval(response.strip())
                if isinstance(segments, list) and segments:
                    logger.info(f"Segmented review into {len(segments)} parts using LLM")
                    return segments
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Could not parse LLM response as list: {e}")
                return [text]

        return [text]

    except Exception as e:
        logger.error(f"Error in LLM segmentation: {e}")
        return [text]


@st.cache_data
def load_reviews_data():
    """
    Load the noema_data.csv file containing reviews.

    Returns:
        pd.DataFrame: DataFrame with reviews
    """
    data_file = Path(DATA_PATH) / "noema_data.csv"

    if not data_file.exists():
        st.error(f"Data file not found: {data_file}")
        logger.error(f"Data file not found: {data_file}")
        return None

    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} reviews from {data_file}")

        # Check if 'review' column exists
        if 'review' not in df.columns:
            st.error(f"'review' column not found in dataset. Available columns: {df.columns.tolist()}")
            logger.error(f"'review' column not found. Columns: {df.columns.tolist()}")
            return None

        # Drop rows with missing reviews
        df = df.dropna(subset=['review'])
        df = df[df['review'].str.strip() != ""]

        logger.info(f"Loaded {len(df)} valid reviews")
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def get_sheet_helper():
    """
    Initialize and cache the SheetHelper for Google Sheets connection.

    Returns:
        SheetHelper: Configured SheetHelper instance
    """
    try:
        sheet_helper = SheetHelper(sheet_url=SHEET_URL, sheet_id=0)
        logger.info("Successfully connected to Google Sheets")
        return sheet_helper
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        logger.error(f"Google Sheets connection error: {e}")
        return None


def preprocess_review(review_text, segmentation_method="none", embedder=None, nlp=None,
                     threshold=0.3, llm_model="llama3.1:8b"):
    """
    Preprocessing pipeline with optional aspect-based segmentation.

    Args:
        review_text (str): Raw review text
        segmentation_method (str): Method to use ("none", "sentence_transformer", "spacy", "llm")
        embedder: SentenceTransformer model (for sentence_transformer method)
        nlp: spaCy model (for spacy method)
        threshold (float): Similarity threshold for sentence_transformer method
        llm_model (str): Ollama model name for LLM method

    Returns:
        list[str]: List of text segments (or single item if no segmentation)
    """
    if not review_text or not isinstance(review_text, str):
        return [review_text]

    # Apply segmentation based on selected method
    if segmentation_method == "sentence_transformer" and embedder is not None:
        segments = segment_review_sentence_transformer(review_text, embedder, threshold)
    elif segmentation_method == "spacy" and nlp is not None:
        segments = segment_review_spacy(review_text, nlp)
    elif segmentation_method == "llm":
        segments = segment_review_llm(review_text, llm_model)
    else:
        segments = [review_text]

    return segments


def get_random_review(df, segmentation_method="none", embedder=None, nlp=None,
                     threshold=0.3, llm_model="llama3.1:8b"):
    """
    Get a random review from the dataframe and optionally segment it.

    Args:
        df (pd.DataFrame): DataFrame containing reviews
        segmentation_method (str): Segmentation method to use
        embedder: SentenceTransformer model
        nlp: spaCy model
        threshold (float): Similarity threshold
        llm_model (str): Ollama model name

    Returns:
        tuple: (original_review, list of segments)
    """
    if df is None or len(df) == 0:
        return None, [None]

    random_idx = np.random.randint(0, len(df))
    review = df.iloc[random_idx]['review']

    # Apply preprocessing pipeline with segmentation
    segments = preprocess_review(
        review,
        segmentation_method=segmentation_method,
        embedder=embedder,
        nlp=nlp,
        threshold=threshold,
        llm_model=llm_model
    )

    return review, segments


def save_to_sheets(sheet_helper, review, classification, segmenter):
    """
    Save the review, classification, and segmentation method to Google Sheets.

    Args:
        sheet_helper (SheetHelper): SheetHelper instance
        review (str): Review text
        classification (str): Classification label
        segmenter (str): Segmentation method used

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        sheet_helper.append_row([review, classification, segmenter])
        logger.info(f"Saved to Google Sheets: {review[:50]}... -> {classification} (segmenter: {segmenter})")
        return True
    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {e}")
        logger.error(f"Failed to save to Google Sheets: {e}")
        return False


def main():
    """
    Main Streamlit app function.
    """
    st.set_page_config(
        page_title="GP Review Human Labeller",
        page_icon="📝",
        layout="wide"
    )

    # Header
    st.title("📝 GP Review Human Labeller")
    st.write("Classify GP reviews into topic categories with aspect-based segmentation")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Segmentation Settings")

        segmentation_method = st.selectbox(
            "Segmentation Method",
            options=["none", "sentence_transformer", "spacy", "llm"],
            index=0,
            help="""
            - **none**: No segmentation, classify entire review
            - **sentence_transformer**: Use AI embeddings to detect topic shifts
            - **spacy**: Use NLP to detect aspect boundaries
            - **llm**: Use Ollama LLM for intelligent segmentation
            """
        )

        # Show threshold slider only for sentence_transformer method
        if segmentation_method == "sentence_transformer":
            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Lower values = more segments (more sensitive to topic shifts)"
            )
        else:
            threshold = 0.3

        # LLM model selection for LLM method
        if segmentation_method == "llm":
            llm_model = st.selectbox(
                "Ollama Model",
                options=["llama3.1:8b", "qwen3:8b", "gpt-oss:20b"],
                index=2
            )
        else:
            llm_model = "gpt-oss:20b"

        st.divider()
        st.caption("💡 Segmentation splits multi-aspect reviews into focused segments for more accurate classification")

        # Display current segmentation info
        if 'current_segmentation_method' in st.session_state:
            st.info(f"**Active Method:** {st.session_state.current_segmentation_method}")
            if st.session_state.current_segmentation_method == "sentence_transformer":
                st.caption(f"Threshold: {st.session_state.current_threshold}")
            elif st.session_state.current_segmentation_method == "llm":
                st.caption(f"Model: {st.session_state.current_llm_model}")

    # Load data
    df = load_reviews_data()

    if df is None:
        st.stop()

    # Load segmentation models
    embedder, nlp = load_segmentation_models()

    # Initialize SheetHelper
    sheet_helper = get_sheet_helper()

    if sheet_helper is None:
        st.stop()

    # Initialize session state
    if 'current_review_original' not in st.session_state:
        original, segments = get_random_review(
            df, segmentation_method, embedder, nlp, threshold, llm_model
        )
        st.session_state.current_review_original = original
        st.session_state.current_review_segments = segments
        st.session_state.current_segment_idx = 0
        st.session_state.current_segmentation_method = segmentation_method
        st.session_state.current_threshold = threshold
        st.session_state.current_llm_model = llm_model

    if 'reviews_labeled' not in st.session_state:
        st.session_state.reviews_labeled = 0

    if 'segments_labeled' not in st.session_state:
        st.session_state.segments_labeled = 0

    # Check if segmentation settings changed - if so, re-segment current review
    settings_changed = (
        st.session_state.get('current_segmentation_method') != segmentation_method or
        st.session_state.get('current_threshold') != threshold or
        st.session_state.get('current_llm_model') != llm_model
    )

    if settings_changed and st.session_state.current_review_original:
        # Re-segment the current review with new settings
        segments = preprocess_review(
            st.session_state.current_review_original,
            segmentation_method=segmentation_method,
            embedder=embedder,
            nlp=nlp,
            threshold=threshold,
            llm_model=llm_model
        )
        st.session_state.current_review_segments = segments
        st.session_state.current_segment_idx = 0
        st.session_state.current_segmentation_method = segmentation_method
        st.session_state.current_threshold = threshold
        st.session_state.current_llm_model = llm_model
        logger.info(f"Re-segmented current review with new settings: {segmentation_method}")

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        st.metric("Reviews Labeled", st.session_state.reviews_labeled)
    with col3:
        st.metric("Segments Labeled", st.session_state.segments_labeled)
    with col4:
        num_segments = len(st.session_state.current_review_segments)
        current_idx = st.session_state.current_segment_idx + 1
        st.metric("Current Segment", f"{current_idx}/{num_segments}")

    st.divider()

    # Always display the full original review
    st.markdown("📄 Full Original Review")
    with st.container():
        st.info(st.session_state.current_review_original)




    # Create bordered container using custom CSS
    st.markdown(
        """
        <style>
        .review-container {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            background-color: #f9f9f9;
            margin: 10px 0px;
        }
        .segment-item {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background-color: #f9f9f9;
            margin: 8px 0px;
        }
        .segment-item-active {
            border: 3px solid #4CAF50;
            border-radius: 8px;
            padding: 15px;
            background-color: #e8f5e9;
            margin: 8px 0px;
            box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3);
        }
        .segment-label {
            background-color: #2196F3;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
        }
        .segment-label-active {
            background-color: #4CAF50;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
        }
        .segment-text {
            margin-top: 8px;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display all segments
    for idx, segment in enumerate(st.session_state.current_review_segments):
        is_current = idx == st.session_state.current_segment_idx

        # Create visual indicator
        if is_current:
            st.markdown(f"**🎯 Segment {idx + 1}/{len(st.session_state.current_review_segments)} - CURRENT**")
            with st.container():
                st.success(segment)
        else:
            st.markdown(f"Segment {idx + 1}/{len(st.session_state.current_review_segments)}")
            with st.container():
                st.warning(segment)

    st.divider()


    current_segment = st.session_state.current_review_segments[st.session_state.current_segment_idx]
    with st.container():
        st.markdown(
            f'<div class="review-container"><h3>{current_segment}</h3></div>',
            unsafe_allow_html=True
        )

    st.divider()


    # Create list of category labels for selectbox
    category_options = [f"{key}: {value}" for key, value in TOPIC_LABELS.items()]

    # Selectbox for category selection
    selected_category = st.selectbox(
        "Select the category that best matches this review:",
        options=category_options,
        index=0,
        help="Choose one category that best describes the main topic of the review"
    )

    # Extract just the label (without the number prefix)
    selected_label = selected_category.split(": ", 1)[1]

    st.divider()

    # Action buttons
    col2, col3, col4 = st.columns([1, 1, 1])

    with col2:
        if st.button("✅ Save & Next Segment", type="primary", use_container_width=True):
            # Save current segment to Google Sheets with segmentation method
            current_segment = st.session_state.current_review_segments[st.session_state.current_segment_idx]
            success = save_to_sheets(
                sheet_helper,
                current_segment,
                selected_label,
                segmentation_method  # Add segmentation method
            )

            if success:
                st.success(f"✅ Saved: **{selected_label}** (Method: {segmentation_method})")
                st.session_state.segments_labeled += 1

                # Check if there are more segments in this review
                if st.session_state.current_segment_idx < len(st.session_state.current_review_segments) - 1:
                    # Move to next segment
                    st.session_state.current_segment_idx += 1
                else:
                    # All segments done, get new review
                    st.session_state.reviews_labeled += 1
                    original, segments = get_random_review(
                        df, segmentation_method, embedder, nlp, threshold, llm_model
                    )
                    st.session_state.current_review_original = original
                    st.session_state.current_review_segments = segments
                    st.session_state.current_segment_idx = 0
                    st.session_state.current_segmentation_method = segmentation_method
                    st.session_state.current_threshold = threshold
                    st.session_state.current_llm_model = llm_model

                # Rerun to show next segment/review
                st.rerun()
            else:
                st.error("❌ Failed to save. Please try again.")

    with col3:
        if st.button("⏭️ Skip Segment", use_container_width=True):
            # Check if there are more segments in this review
            if st.session_state.current_segment_idx < len(st.session_state.current_review_segments) - 1:
                # Move to next segment
                st.session_state.current_segment_idx += 1
            else:
                # All segments done, get new review
                original, segments = get_random_review(
                    df, segmentation_method, embedder, nlp, threshold, llm_model
                )
                st.session_state.current_review_original = original
                st.session_state.current_review_segments = segments
                st.session_state.current_segment_idx = 0
                st.session_state.current_segmentation_method = segmentation_method
                st.session_state.current_threshold = threshold
                st.session_state.current_llm_model = llm_model
            st.rerun()

    with col4:
        if st.button("🔄 Skip Entire Review", use_container_width=True):
            # Get new review regardless of segments
            original, segments = get_random_review(
                df, segmentation_method, embedder, nlp, threshold, llm_model
            )
            st.session_state.current_review_original = original
            st.session_state.current_review_segments = segments
            st.session_state.current_segment_idx = 0
            st.session_state.current_segmentation_method = segmentation_method
            st.session_state.current_threshold = threshold
            st.session_state.current_llm_model = llm_model
            st.rerun()

    # Footer
    st.divider()
    st.caption(f"🔗 Connected to Google Sheets")
    st.caption(f"💾 Data source: {DATA_PATH}/noema_data.csv")

    # Info box
    with st.expander("ℹ️ Instructions & Help", expanded=False):
        st.markdown("""
        ### How to use this app:

        1. **Configure segmentation** in the sidebar (optional)
        2. **Read the review segment** displayed in the green-bordered box
        3. **Select the appropriate category** from the dropdown menu
        4. **Click "Save & Next Segment"** to save and move to next segment/review
        5. **Click "Skip Segment"** to skip current segment
        6. **Click "Skip Entire Review"** to skip all remaining segments

        ### About Segmentation Methods:

        - **None**: Classify the entire review as one unit
        - **SentenceTransformer**: Uses AI embeddings to detect semantic topic shifts
          - Adjust threshold: Lower = more segments
        - **Spacy**: Uses NLP to detect linguistic boundaries and contrast markers
        - **LLM**: Uses Ollama to intelligently segment reviews by aspects

        ### About Categories:

        The 16 topic categories cover various aspects of GP practice feedback:
        - Appointments & Access
        - Clinical Care & Treatment
        - Staff Interactions
        - Administrative Processes
        - Facilities & Services
        - Privacy & Confidentiality
        - And more...

        ### Segmentation Benefits:

        Multi-aspect reviews (e.g., "The doctor was great but the wait time was terrible")
        are automatically split so each aspect can be classified separately, leading to
        more accurate and granular feedback analysis.
        """)


if __name__ == "__main__":
    main()
