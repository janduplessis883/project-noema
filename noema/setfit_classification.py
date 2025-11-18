"""
SetFit Classification Script
----------------------------
Performs classification on real GP review data using trained SetFit models.
Loads the saved model and applies it to reviews from noema_data.csv.
"""

import pandas as pd
import numpy as np
from setfit import SetFitModel
from pathlib import Path
from loguru import logger
from typing import Optional, List, Dict
import sys

# Import params for paths
from noema.params import DATA_PATH, MODEL_PATH

logger.add("setfit_classification.log", rotation="5000 KB")

# Topic mapping - matches training pipeline
TOPIC_LABELS = {
    0: "Appointment Availability & Wait Time (Remote/On-Site)",
    1: "Telephone & Digital Access & Services",
    2: "Administrative Efficiency & Process",
    3: "Accessibility & Facilities",
    4: "Prescription Management",
    5: "Diagnosis & Treatment Quality",
    6: "Test Results & Follow-up",
    7: "Clinical Explanation & Health Information",
    8: "Specialized Health Services",
    9: "Medication Safety & Accuracy",
    10: "Clinical Staff Interpersonal Skills & Attitude",
    11: "Administrative Staff Interpersonal Skills & Attitude",
    12: "Continuity of Care",
    13: "Privacy & Confidentiality",
    14: "Feedback & Complaints Process",
}

SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive"
}


def load_data(csv_path: str, review_column: str = "review") -> pd.DataFrame:
    """
    Loads the review data from CSV file.

    Args:
        csv_path: Path to the CSV file
        review_column: Name of the column containing review text

    Returns:
        pd.DataFrame: DataFrame with reviews
    """
    logger.info(f"Loading data from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows")

        # Check if review column exists
        if review_column not in df.columns:
            logger.error(f"Column '{review_column}' not found in CSV")
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Column '{review_column}' not found")

        # Drop rows with missing reviews
        original_len = len(df)
        df = df.dropna(subset=[review_column])
        if len(df) < original_len:
            logger.warning(f"Dropped {original_len - len(df)} rows with missing reviews")

        # Drop empty reviews
        df = df[df[review_column].str.strip() != ""]
        logger.info(f"Final dataset size: {len(df)} reviews")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_model(model_path: str) -> SetFitModel:
    """
    Loads a trained SetFit model from disk.

    Args:
        model_path: Path to the saved model directory

    Returns:
        SetFitModel: Loaded model
    """
    logger.info(f"Loading model from: {model_path}")

    try:
        model = SetFitModel.from_pretrained(model_path)
        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def classify_reviews(
    model: SetFitModel,
    reviews: List[str],
    batch_size: int = 32
) -> np.ndarray:
    """
    Classifies a list of reviews using the SetFit model.

    Args:
        model: Trained SetFit model
        reviews: List of review texts
        batch_size: Number of reviews to process at once

    Returns:
        np.ndarray: Array of predicted labels
    """
    logger.info(f"Classifying {len(reviews)} reviews in batches of {batch_size}")

    all_predictions = []

    # Process in batches for efficiency
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        predictions = model.predict(batch)
        all_predictions.extend(predictions)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch)}/{len(reviews)} reviews")

    logger.info("Classification completed")
    return np.array(all_predictions)


def add_predictions_to_dataframe(
    df: pd.DataFrame,
    predictions: np.ndarray,
    task: str = "sentiment"
) -> pd.DataFrame:
    """
    Adds prediction results to the dataframe.

    Args:
        df: Original dataframe
        predictions: Array of predicted labels
        task: Type of classification ("sentiment" or "topic")

    Returns:
        pd.DataFrame: DataFrame with predictions added
    """
    df = df.copy()

    if task == "sentiment":
        df['predicted_sentiment'] = predictions
        df['sentiment_label'] = df['predicted_sentiment'].map(SENTIMENT_LABELS)

    elif task == "topic":
        df['predicted_topic'] = predictions
        df['topic_label'] = df['predicted_topic'].map(TOPIC_LABELS)

    return df


def generate_classification_report(df: pd.DataFrame, task: str = "sentiment") -> Dict:
    """
    Generates a summary report of the classification results.

    Args:
        df: DataFrame with predictions
        task: Type of classification

    Returns:
        Dict: Summary statistics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Classification Report - {task.upper()}")
    logger.info(f"{'='*60}")

    report = {
        'total_reviews': len(df),
        'task': task
    }

    if task == "sentiment":
        sentiment_counts = df['sentiment_label'].value_counts()
        logger.info(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"  {sentiment}: {count} ({pct:.1f}%)")

        report['sentiment_distribution'] = sentiment_counts.to_dict()

    elif task == "topic":
        topic_counts = df['topic_label'].value_counts()
        logger.info(f"\nTopic Distribution:")
        for topic, count in topic_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"  {topic}: {count} ({pct:.1f}%)")

        report['topic_distribution'] = topic_counts.to_dict()

        # Show top 5 topics
        logger.info(f"\nTop 5 Topics:")
        for i, (topic, count) in enumerate(topic_counts.head(5).items(), 1):
            pct = (count / len(df)) * 100
            logger.info(f"  {i}. {topic}: {count} ({pct:.1f}%)")

    logger.info(f"\n{'='*60}\n")

    return report


def save_results(
    df: pd.DataFrame,
    output_path: str,
    task: str = "sentiment"
):
    """
    Saves the classification results to CSV.

    Args:
        df: DataFrame with predictions
        output_path: Path to save the results
        task: Type of classification
    """
    logger.info(f"Saving results to: {output_path}")

    # Select relevant columns based on task
    if task == "sentiment":
        output_columns = ['review', 'predicted_sentiment', 'sentiment_label']
    elif task == "topic":
        output_columns = ['review', 'predicted_topic', 'topic_label']
    else:
        # Default fallback
        output_columns = ['review']

    # Include any other columns from original data
    for col in df.columns:
        if col not in output_columns and col != 'review':
            output_columns.append(col)

    # Save to CSV
    df[output_columns].to_csv(output_path, index=False)
    logger.info(f"Results saved successfully ({len(df)} rows)")


def classify_dataset(
    task: str = "sentiment",
    review_column: str = "review",
    output_filename: Optional[str] = None,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Main function to classify the noema dataset.

    Args:
        task: Type of classification ("sentiment" or "topic")
        review_column: Name of the column containing review text
        output_filename: Optional custom output filename
        batch_size: Batch size for classification

    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting SetFit Classification Pipeline - {task.upper()}")
    logger.info(f"{'='*60}\n")

    # Paths
    data_file = Path(DATA_PATH) / "noema_data.csv"
    model_dir = Path(MODEL_PATH) / f"setfit_gp_{task}"

    if output_filename is None:
        output_filename = f"noema_classified_{task}.csv"
    output_path = Path(DATA_PATH) / output_filename

    # Validate paths
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.info(f"Expected model at: {model_dir}")
        logger.info("Please train the model first using setfit_pipe.py")
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load data
    df = load_data(str(data_file), review_column=review_column)

    # Load model
    model = load_model(str(model_dir))

    # Get reviews as list
    reviews = df[review_column].tolist()

    # Classify reviews
    predictions = classify_reviews(model, reviews, batch_size=batch_size)

    # Add predictions to dataframe
    df_with_predictions = add_predictions_to_dataframe(df, predictions, task=task)

    # Generate report
    report = generate_classification_report(df_with_predictions, task=task)

    # Save results
    save_results(df_with_predictions, str(output_path), task=task)

    logger.info(f"\n{'='*60}")
    logger.info("Classification pipeline completed successfully!")
    logger.info(f"{'='*60}\n")

    return df_with_predictions


def main():
    """
    Main entry point for the classification script.
    """
    # You can modify these parameters as needed
    task = "topic"  # Change to "sentiment" for sentiment classification
    review_column = "review"  # Name of the column containing review text

    try:
        # Run classification
        df_results = classify_dataset(
            task=task,
            review_column=review_column,
            batch_size=32
        )

        # Optional: Display some example predictions
        logger.info("\nSample Predictions (first 5 reviews):")
        logger.info("-" * 60)
        for idx in range(min(5, len(df_results))):
            review = df_results.iloc[idx][review_column]
            if task == "sentiment":
                prediction = df_results.iloc[idx]['sentiment_label']
            else:
                prediction = df_results.iloc[idx]['topic_label']

            logger.info(f"\nReview: {review[:100]}...")
            logger.info(f"Prediction: {prediction}")

        logger.info("\n" + "="*60)
        logger.info("Done! Check the output CSV file for full results.")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
