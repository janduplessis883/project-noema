"""
SetFit Training Pipeline for GP Surgery Reviews
------------------------------------------------
This script trains a SetFit model for sentiment classification and
topic classification on synthetic GP Surgery reviews.

Updated with 20 comprehensive categories based on GP feedback taxonomy.
"""

import pandas as pd
import numpy as np
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
from typing import Tuple, Dict
from noema.params import DATA_PATH, MODEL_PATH

logger.add("setfit_training.log", rotation="5000 KB")

# Topic mapping - 15 categories
TOPIC_LABELS = {
    0: "Appointment Availability & Access",
    1: "Online Services",
    2: "Admin Processes",
    3: "Facilities & Cleanliness",
    4: "Prescription & Medication Management",
    5: "Clinical Treatment Quality",
    6: "Test Results & Follow-up",
    7: "Health Promotion",
    8: "Nursing & HCA Services",
    9: "Reception Staff Interaction",
    10: "Clinical Staff Profesionalism & Attitude",
    11: "Continuity of Care",
    12: "Privacy & Confidentiality",
    13: "Feedback & Complaints Process",
    14: "Overall Experience",
}

# 🅾️ Load human-labeled training dataset
training_dataset_path = Path(DATA_PATH) / "training_dataset_v6.csv"
logger.info(f"Loading training dataset from: {training_dataset_path}")





def create_synthetic_gp_reviews():
    """
    Creates a synthetic dataset of GP Surgery reviews with sentiment labels.

    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """

    # Positive reviews
    positive_reviews = [
        "Excellent service, the receptionist was very helpful and kind",
        "The doctor took time to listen to my concerns and explained everything clearly",
        "Quick appointment booking online, very convenient system",
        "Clean and modern surgery with friendly staff throughout",
        "GP was thorough and professional, felt well cared for",
        "Nurse was compassionate and made me feel at ease during vaccination",
        "Phone consultation was efficient and the doctor called exactly on time",
        "Prescription ready quickly at the pharmacy, no issues",
        "Easy to get urgent appointment when needed, staff very responsive",
        "Practice manager resolved my complaint promptly and professionally",
        "Blood test appointment ran smoothly, phlebotomist was skilled",
        "Mental health support from GP was understanding and helpful",
        "Child's immunisation went well, nurse was gentle and reassuring",
        "Diabetic review was comprehensive, felt supported with my condition",
        "Fantastic doctor who really cares about patients",
        "Reception staff always greet with a smile",
        "Modern facility with good accessibility for wheelchairs",
        "Online repeat prescription system works perfectly",
        "GP followed up after hospital discharge which was reassuring",
        "Practice website has clear information and helpful resources",
        "Nurse practitioner was knowledgeable about chronic conditions",
        "Appointment reminder texts are very useful",
        "Longer appointments available when needed to discuss multiple issues",
        "Doctor explained test results in terms I could understand",
        "Efficient triage system ensures urgent cases seen quickly",
        "Pleasant waiting room with good distraction for children",
        "Continuity of care - able to see same doctor for ongoing issues",
        "Staff wear name badges which is helpful and professional",
        "Good parking facilities available near the surgery",
        "Practice open early for working patients which is helpful",
        "Doctor took my symptoms seriously and arranged referral",
        "Blood pressure monitoring service very convenient",
        "Helpful advice from pharmacist about medication side effects",
        "Practice nurse explained wound care instructions thoroughly",
        "Flexible appointment times including evenings",
        "Doctor reviewed all my medications and made helpful adjustments",
        "Cervical screening done sensitively with clear explanation",
        "Asthma review comprehensive with peak flow measurements",
        "Travel vaccination service excellent with good advice",
        "Doctor showed empathy and understanding during difficult time",
    ]

    # Negative reviews
    negative_reviews = [
        "Impossible to get through on phone, waited 45 minutes",
        "Rude receptionist made me feel like a burden",
        "Had to wait 3 weeks for routine appointment, unacceptable",
        "Doctor seemed rushed and didn't listen to my concerns",
        "Prescription not ready despite ordering days ago",
        "Dirty waiting room, magazines from years ago",
        "Online booking system never has appointments available",
        "Called about test results, no one bothered to ring back",
        "GP dismissed my symptoms without proper examination",
        "Reception staff discussing patients loudly, no privacy",
        "Appointment cancelled last minute with no explanation",
        "Sent to wrong location for blood test, very frustrating",
        "Doctor looked at computer screen entire time, no eye contact",
        "Couldn't get urgent appointment when genuinely needed one",
        "Phone rang 30 times before anyone answered",
        "Repeat prescription request lost twice",
        "Very long wait past appointment time with no apology",
        "Practice never answers emails sent through online system",
        "Referral delayed for months despite being urgent",
        "Patronising attitude from doctor who didn't take me seriously",
        "No disabled access despite being medical facility",
        "Told different information by different staff members",
        "Home visit request refused despite being housebound",
        "Medical records request took forever to process",
        "Doctor prescribed without examining properly",
        "Unable to see same GP, always different doctor",
        "Appointment system favours certain patients unfairly",
        "Staff shortage means limited appointments available",
        "Practice manager unhelpful and defensive about complaints",
        "Blood test results took weeks to get back",
        "No interpreter available despite requesting in advance",
        "Medication review felt rushed and box-ticking exercise",
        "Nurse seemed inexperienced and unsure about procedure",
        "Car park always full, difficult for elderly patients",
        "Doctor interrupted me constantly, wouldn't let me finish explaining",
        "Mental health concerns brushed aside with leaflet",
        "Test results given over phone in public area, no privacy",
        "Reception area too small, patients standing in corridor",
        "Website information outdated and contradicts phone messages",
        "Charged for letter when should have been free on NHS",
    ]

    # Create balanced dataset
    data = {
        'text': positive_reviews + negative_reviews,
        'label': [1] * len(positive_reviews) + [0] * len(negative_reviews)
    }

    df = pd.DataFrame(data)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Created synthetic dataset with {len(df)} reviews")
    logger.info(f"Positive reviews: {sum(df['label'] == 1)}")
    logger.info(f"Negative reviews: {sum(df['label'] == 0)}")

    return df


def load_human_labeled_dataset() -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Loads the human-labeled training dataset from CSV.

    The CSV should have columns:
    - 'review': The review text
    - 'class': The topic class label (text)
    - 'segmenter': The segmentation method used (ignored)

    Returns:
        Tuple of (DataFrame, topic_mapping)
    """

    training_dataset_path = Path(DATA_PATH) / "training_dataset_v6.csv"

    if not training_dataset_path.exists():
        logger.error(f"Training dataset not found: {training_dataset_path}")
        raise FileNotFoundError(f"Training dataset not found: {training_dataset_path}")

    # Load the CSV
    df = pd.read_csv(training_dataset_path)
    logger.info(f"Loaded {len(df)} training examples from {training_dataset_path}")

    # Check required columns
    if 'review' not in df.columns or 'class' not in df.columns:
        logger.error(f"CSV must have 'review' and 'class' columns. Found: {df.columns.tolist()}")
        raise ValueError(f"CSV must have 'review' and 'class' columns")

    # Create reverse mapping from class name to ID
    class_name_to_id = {v: k for k, v in TOPIC_LABELS.items()}

    # Map class names to numeric IDs
    df['label'] = df['class'].map(class_name_to_id)

    # Check for unmapped labels
    unmapped = df[df['label'].isna()]
    if len(unmapped) > 0:
        logger.warning(f"Found {len(unmapped)} rows with unmapped class labels:")
        logger.warning(f"Unique unmapped labels: {unmapped['class'].unique().tolist()}")
        logger.warning("These rows will be dropped")
        df = df.dropna(subset=['label'])

    # Convert label to int
    df['label'] = df['label'].astype(int)

    # Rename 'review' to 'text' for consistency with training pipeline
    df = df.rename(columns={'review': 'text'})

    # Add topic_name column
    df['topic_name'] = df['label'].map(TOPIC_LABELS)

    # Keep only required columns
    df = df[['text', 'label', 'topic_name']].copy()

    logger.info(f"Successfully mapped {len(df)} training examples")
    logger.info(f"Number of unique topics: {df['label'].nunique()}")
    logger.info(f"Topics distribution:\n{df['label'].value_counts().sort_index()}")

    # Check if all topics are represented
    missing_topics = set(TOPIC_LABELS.keys()) - set(df['label'].unique())
    if missing_topics:
        logger.warning(f"Warning: The following topics have no training examples: {missing_topics}")
        for topic_id in missing_topics:
            logger.warning(f"  {topic_id}: {TOPIC_LABELS[topic_id]}")

    return df, TOPIC_LABELS


def create_topic_classification_dataset_synthetic() -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Creates a synthetic dataset for topic classification of GP reviews.

    NOTE: This function is kept for reference but should not be used.
    Use load_human_labeled_dataset() instead.

    15 Topics based on comprehensive GP practice feedback taxonomy.

    Returns:
        Tuple of (DataFrame, topic_mapping)
    """
    logger.warning("Using synthetic data - this should only be used for testing!")

    # Category 0: Appointment Availability
    appointment_reviews = [
        "Can't get a routine appointment for weeks",
        "Same-day appointments available when I needed one urgently",
        "Had to wait 3 weeks just for a routine check-up",
        "Emergency slots always available which is reassuring",
        "Impossible to book appointments nowadays, system is broken",
        "Quick appointment booking, got seen within days",
        "Urgent care appointments easy to access",
        "Long waits for routine appointments are frustrating",
        "Appointment availability has improved recently",
        "No appointments available for my urgent issue",
        "Booking system shows no slots for next month",
        "Got appointment within 48 hours, excellent availability",
        "Five week wait for non-urgent matters is too long",
        "Walk-in clinic hours very useful for busy people",
        "Appointment slots fill up within minutes of release",
        "Extended hours appointments perfect for working patients",
        "Pre-bookable appointments always fully booked",
        "Emergency appointment system works really well",
        "Reduced appointment availability is concerning",
        "Saturday morning appointments very convenient",
        "Never enough appointments despite patient demand",
        "Telephone triage helps prioritize urgent cases",
        "Waiting times for appointments keep increasing",
        "Appointment booking improved with new system",
        "Cannot get appointment when genuinely needed",
        "Good range of appointment types available",
        "Frustrating lack of available slots",
        "Nurse appointments easier to book than GP",
        "Advanced booking system not fit for purpose",
        "Regular slots for chronic disease management helpful",
    ]

    # Category 1: Telephone & Digital Access
    phone_digital_reviews = [
        "Phone lines always engaged, impossible to get through",
        "Online eConsult system works brilliantly",
        "Waited 45 minutes on hold before someone answered",
        "Digital booking app is very user-friendly",
        "Phone system needs serious improvement",
        "Online forms make accessing care much easier",
        "Can never get through when calling in the morning",
        "Website booking system is convenient",
        "Telephone access is terrible, constantly engaged",
        "eConsult response received same day, very efficient",
    ]

    # Category 2: Administrative Efficiency
    admin_efficiency_reviews = [
        "Referral processed quickly and efficiently",
        "Medical records request took forever",
        "Registration was smooth and professional",
        "Administrative errors in my patient file",
        "Sick notes ready promptly when needed",
        "Lost my repeat prescription request twice",
        "Efficient handling of all admin tasks",
        "Paperwork always delayed or incorrect",
        "Staff very competent with administrative duties",
        "Complete chaos with admin, wrong info given",
    ]

    # Category 3: Accessibility & Facilities
    accessibility_reviews = [
        "Excellent disabled access with ramps and automatic doors",
        "Car park always full, difficult for elderly",
        "Building is clean and well-maintained",
        "No disabled parking despite being medical facility",
        "Modern waiting room with good seating",
        "Waiting area too small, people standing in corridor",
        "Good parking and wheelchair access throughout",
        "Premises look neglected and run down",
        "Bright, clean, and comfortable facilities",
        "Poor physical access for those with mobility issues",
    ]

    # Category 4: Prescription Management
    prescription_reviews = [
        "Repeat prescriptions ready on time every month",
        "Prescription not ready despite ordering days ago",
        "Electronic prescription system very convenient",
        "Wrong medication dispensed, very concerning",
        "Online ordering works perfectly",
        "Prescription request lost multiple times",
        "Efficient prescription collection service",
        "Always problems with repeat prescriptions",
        "Prescription delivery service is excellent",
        "Medication never ready when promised",
    ]

    # Category 5: Diagnosis & Treatment Quality
    diagnosis_treatment_reviews = [
        "Doctor made accurate diagnosis and effective treatment",
        "Misdiagnosed twice, symptoms got worse",
        "Thorough examination and appropriate referral",
        "GP prescribed without proper examination",
        "Evidence-based treatment approach, very professional",
        "Treatment didn't work, no follow-up offered",
        "Competent clinical care from experienced doctor",
        "Poor quality medical advice given",
        "Doctor very knowledgeable about my condition",
        "Inadequate treatment for serious symptoms",
    ]

    # Category 6: Test Results & Follow-up
    test_followup_reviews = [
        "Test results communicated promptly and clearly",
        "Never called back about blood test results",
        "Follow-up care arranged appropriately",
        "No follow-up after hospital discharge",
        "Results available online quickly",
        "Had to chase test results multiple times",
        "Doctor followed up as promised",
        "Test results lost in the system",
        "Clear explanation of what results mean",
        "Weeks passed with no results communication",
    ]

    # Category 7: Clinical Explanation & Information
    explanation_reviews = [
        "Doctor explained diagnosis in terms I understood",
        "No explanation of treatment options given",
        "Clear information about medication and side effects",
        "Didn't explain what was wrong with me",
        "Nurse took time to explain procedure thoroughly",
        "Rushed consultation with unclear information",
        "Detailed explanation of self-management strategies",
        "Left confused about what to do next",
        "Treatment plan explained comprehensively",
        "No guidance on lifestyle changes needed",
    ]

    # Category 8: Specialized Health Services
    specialized_reviews = [
        "Flu vaccination service well-organized",
        "Long wait for travel vaccinations",
        "Mental health support very understanding",
        "Mental health concerns dismissed",
        "Screening program efficient and professional",
        "Vaccination service could be better organized",
        "Excellent support for mental wellbeing",
        "No proper mental health pathway available",
        "Immunisation program runs smoothly",
        "Vaccination appointments poorly coordinated",
    ]

    # Category 9: Medication Safety & Accuracy
    medication_safety_reviews = [
        "Medication review highlighted important interactions",
        "Wrong dosage prescribed, dangerous error",
        "Pharmacist caught potential medication error",
        "Given medication I'm allergic to",
        "Thorough check of all medications for safety",
        "Prescription errors keep happening",
        "Doctor reviewed medications carefully",
        "Concerning lack of medication safety checks",
        "Proper discussion of medication risks",
        "Medication errors not taken seriously",
    ]

    # Category 10: Clinical Staff Interpersonal Skills
    clinical_interpersonal_reviews = [
        "Doctor listened carefully and showed genuine concern",
        "GP seemed rushed and uninterested",
        "Nurse very empathetic during difficult consultation",
        "Doctor didn't listen to my concerns properly",
        "Felt truly cared for by compassionate clinician",
        "Cold and clinical approach lacking warmth",
        "Adequate time given to discuss all my worries",
        "Felt rushed through appointment",
        "Doctor's empathy made difficult news easier",
        "No emotional support offered during consultation",
    ]

    # Category 11: Administrative Staff Attitude
    admin_staff_reviews = [
        "Receptionist always friendly and helpful",
        "Rude reception staff made me feel unwelcome",
        "Admin team courteous and professional",
        "Front desk staff dismissive and unhelpful",
        "Reception greeted me with a smile",
        "Receptionists discussing patients loudly, unprofessional",
        "Polite and efficient service at reception",
        "Receptionist made me feel like a burden",
        "Helpful staff at front desk",
        "Disrespectful attitude from admin team",
    ]

    # Category 12: Continuity of Care
    continuity_reviews = [
        "Can always see my regular GP",
        "See different doctor every time, no continuity",
        "Excellent coordination with hospital",
        "No coordination between appointments",
        "My usual doctor knows my history well",
        "Referral not followed up properly",
        "Consistent care from same clinician",
        "Lack of continuity affecting treatment",
        "Seamless care coordination with specialists",
        "Different GP each time means starting from scratch",
        "Regular GP knows my medical history",
        "Never see the same doctor twice",
        "Good continuity between appointments",
        "Information not shared between doctors",
        "Follow-up care well coordinated",
        "No handover between clinicians",
        "Familiar face at each visit helps",
        "Always explaining history again",
        "Integrated care pathway excellent",
        "Disconnected care experience",
    ]

    # Category 13: Privacy & Confidentiality
    privacy_reviews = [
        "Confidentiality maintained at all times",
        "Reception staff discussing patients where others can hear",
        "Private consultation rooms, felt comfortable",
        "Test results given in public area, no privacy",
        "Data security taken seriously",
        "Overheard personal information at front desk",
        "Discrete and professional privacy maintained",
        "Concerns about confidentiality practices",
        "Respected my need for privacy",
        "Privacy not protected at reception",
        "Medical records kept secure",
        "Personal details visible to other patients",
        "Confidential discussions in private",
        "Lack of privacy at reception desk",
        "GDPR compliance evident",
        "Sensitive information disclosed publicly",
        "Privacy screens used appropriately",
        "Confidentiality breached",
        "Professional handling of personal data",
        "Privacy concerns not addressed",
    ]

    # Category 14: Feedback & Complaints Process
    complaints_reviews = [
        "Complaint handled professionally and promptly",
        "My complaint was ignored completely",
        "Good mechanism for patient feedback",
        "Defensive response to my concerns",
        "Concerns taken seriously and resolved",
        "Complaints procedure not transparent",
        "Practice manager very responsive to feedback",
        "No one wants to deal with complaints",
        "Constructive response to my suggestions",
        "Complaint dismissed without investigation",
        "Easy to provide formal feedback",
        "Complaints process unclear",
        "Listened to my concerns",
        "Feedback not acknowledged",
        "Patient voice valued here",
        "Complaints procedure inadequate",
        "Responsive to patient feedback",
        "No mechanism for complaints",
        "Proactive in addressing issues",
        "Defensive when criticized",
    ]

    # Combine all reviews (15 topics only)
    all_reviews = (
        appointment_reviews + phone_digital_reviews + admin_efficiency_reviews +
        accessibility_reviews + prescription_reviews + diagnosis_treatment_reviews +
        test_followup_reviews + explanation_reviews + specialized_reviews +
        medication_safety_reviews + clinical_interpersonal_reviews +
        admin_staff_reviews + continuity_reviews + privacy_reviews +
        complaints_reviews
    )

    # Create labels (0-14 for 15 topics)
    all_labels = []
    all_labels.extend([0] * len(appointment_reviews))
    all_labels.extend([1] * len(phone_digital_reviews))
    all_labels.extend([2] * len(admin_efficiency_reviews))
    all_labels.extend([3] * len(accessibility_reviews))
    all_labels.extend([4] * len(prescription_reviews))
    all_labels.extend([5] * len(diagnosis_treatment_reviews))
    all_labels.extend([6] * len(test_followup_reviews))
    all_labels.extend([7] * len(explanation_reviews))
    all_labels.extend([8] * len(specialized_reviews))
    all_labels.extend([9] * len(medication_safety_reviews))
    all_labels.extend([10] * len(clinical_interpersonal_reviews))
    all_labels.extend([11] * len(admin_staff_reviews))
    all_labels.extend([12] * len(continuity_reviews))
    all_labels.extend([13] * len(privacy_reviews))
    all_labels.extend([14] * len(complaints_reviews))

    # Create DataFrame
    df = pd.DataFrame({
        'text': all_reviews,
        'label': all_labels,
        'topic_name': [TOPIC_LABELS[label] for label in all_labels]
    })

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Created topic classification dataset with {len(df)} reviews")
    logger.info(f"Number of topics: {df['label'].nunique()}")
    logger.info(f"Topics distribution:\n{df['label'].value_counts().sort_index()}")

    return df, TOPIC_LABELS


def train_setfit_model(df, model_name="sentence-transformers/all-roberta-large-v1",
                       num_epochs=3, batch_size=16):
    """
    Trains a SetFit model on the provided dataset.

    Args:
        df: DataFrame with 'text' and 'label' columns
        model_name: Pre-trained sentence transformer model to use
        num_epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        model: Trained SetFit model
        metrics: Dictionary of evaluation metrics
        test_df: Test DataFrame
    """

    logger.info(f"Training SetFit model with {model_name}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    # Initialize SetFit model
    model = SetFitModel.from_pretrained(model_name)

    # Set up training arguments
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric="accuracy",
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

    # Evaluate on test set
    logger.info("Evaluating model...")
    y_true = test_df['label'].values
    y_pred = model.predict(test_df['text'].tolist())

    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'classification_report': report,
        'confusion_matrix': cm,
        'accuracy': report['accuracy'],
        'test_predictions': y_pred,
        'test_labels': y_true
    }

    return model, metrics, test_df


def plot_confusion_matrix(cm, save_path="confusion_matrix.png", labels=None):
    """Plots and saves confusion matrix."""
    figsize = (14, 12) if cm.shape[0] > 10 else (8, 6)
    plt.figure(figsize=figsize)

    if labels and len(labels) > 10:
        # For many classes, use smaller font
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=labels, yticklabels=labels,
                   annot_kws={"size": 7})
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def save_model(model, model_path="models/setfit_gp_reviews"):
    """Saves the trained SetFit model."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    logger.info(f"Model saved to {model_path}")


def test_model_predictions(model, task="sentiment"):
    """Tests the model with some example predictions."""

    if task == "sentiment":
        test_examples = [
            "The doctor was very understanding and took time to explain everything",
            "Terrible service, waited 2 hours past appointment time",
            "Clean facility with friendly staff all around",
            "Rude receptionist wouldn't help with my query",
            "Quick and efficient prescription service",
        ]
        logger.info("\n=== Model Predictions on Test Examples ===")
        for text in test_examples:
            pred = model.predict([text])[0]
            sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
            logger.info(f"Text: {text}")
            logger.info(f"Prediction: {sentiment} ({pred})\n")
    else:
        test_examples = [
            "Phone lines always busy, took 30 minutes to get through",
            "Prescription ready when I arrived for collection",
            "Waiting room was spotlessly clean",
            "Blood test results explained clearly by doctor",
            "GP took time to listen to all my concerns",
        ]
        logger.info("\n=== Model Predictions on Test Examples ===")
        for text in test_examples:
            pred = model.predict([text])[0]
            topic = TOPIC_LABELS.get(pred, "Unknown")
            logger.info(f"Text: {text}")
            logger.info(f"Prediction: {topic} (ID: {pred})\n")


def main():
    """Main training pipeline."""

    logger.info("=" * 60)
    logger.info("🅾️ SetFit Training Pipeline for GP Surgery Reviews 🅾️")
    logger.info("=" * 60)

    # Choose task: sentiment or topic classification
    task = "topic"  # Change to "sentiment" for sentiment classification

    if task == "sentiment":
        logger.info("\nTask: Sentiment Classification (Positive/Negative)")
        df = create_synthetic_gp_reviews()
        topic_mapping = None
        num_epochs = 2
    else:
        logger.info("\nTask: Topic Classification (15 Categories)")
        logger.info("Loading human-labeled training data...")
        df, topic_mapping = load_human_labeled_dataset()
        num_epochs = 8  # Increased epochs for better accuracy

    # Train the model with optimized hyperparameters
    model, metrics, test_df = train_setfit_model(
        df,
        model_name="sentence-transformers/paraphrase-mpnet-base-v2",
        num_epochs=num_epochs,
        batch_size=16  # Smaller batch size for better generalization
    )

    # Plot confusion matrix
    if task == "topic":
        topic_names = [TOPIC_LABELS[i] for i in range(15)]
        plot_confusion_matrix(metrics['confusion_matrix'],
                            save_path="confusion_matrix_topics.png",
                            labels=topic_names)
    else:
        plot_confusion_matrix(metrics['confusion_matrix'])

    # Save the model
    save_model(model, f"models/setfit_gp_{task}")

    # Test with examples
    test_model_predictions(model, task=task)

    # Save test results
    test_results = test_df.copy()
    test_results['predicted'] = metrics['test_predictions']
    if task == "topic":
        test_results['predicted_topic'] = test_results['predicted'].map(TOPIC_LABELS)
    test_results.to_csv(f"setfit_{task}_test_results.csv", index=False)
    logger.info(f"Test results saved to setfit_{task}_test_results.csv")

    logger.info("\n" + "=" * 60)
    logger.info(f"Training completed! Model accuracy: {metrics['accuracy']:.4f}")
    logger.info("=" * 60)

    return model, metrics


if __name__ == "__main__":
    main()
