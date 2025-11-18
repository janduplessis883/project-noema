from .ollama import ask_ollama
from .params import DATA_PATH
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load data
data = pd.read_csv(f'{DATA_PATH}/noema_data_scored.csv')

# Create a column for scores if it doesn't exist
if 'llm_richness_score' not in data.columns:
    data['llm_richness_score'] = np.nan

# --- MODIFICATION START ---
# Find the index of the first row that still needs a score (i.e., where the score is NaN)
# This handles the restart logic: if all rows are scored, start_index will be len(data).
# Use .first_valid_index() on the inverted condition (isnull()) to find the first NaN.
start_index_series = data['llm_richness_score'].isnull()
# Get the index of the *first* True value (which means the first NaN)
first_nan_index = start_index_series[start_index_series].index.min()
# If all values are non-NaN, first_nan_index will be None.
start_index = first_nan_index if pd.notna(first_nan_index) else len(data)

print(f"Starting score generation from index: {start_index} (Skipping {start_index} already scored entries).")
# --- MODIFICATION END ---

batch_size = 20

# Outer loop over batches, starting from the calculated 'start_index'
for start_idx in range(start_index, len(data), batch_size):
    end_idx = min(start_idx + batch_size, len(data))
    # We use .loc to ensure we are working with the actual index, which is important
    # for assigning the score back to the original DataFrame using .at[i, ...].
    batch = data.loc[start_idx:end_idx-1] # Note: end_idx is exclusive, so use end_idx-1 for .loc slicing to be inclusive up to the last index of the batch

    # Inner loop with tqdm for progress within batch
    # We iterate over the *actual index* (i) and the review text
    for i, review in tqdm(batch['review'].items(), total=len(batch), desc=f"Batch {(start_idx//batch_size) + 1} (Indices {start_idx}-{end_idx-1})"):
        prompt = f"""
        You are rating patient feedback for *richness and value*.
        Rate from 1–5:
        1 = vague or generic,
        2 = somewhat specific - might mention a detail,
        3 = somewhat detailed - no actionable insights,
        4 = detailed and specific - at least one actionable insight,
        5 = highly rich, specific, and multiple actionable insights.
        Return only an integer rating from 1 to 5. No explanation is needed.
        Feedback: "{review}"
        """

        try:
            response = ask_ollama(
                user_prompt=prompt,
                system_prompt="You are a helpful AI assistant that rates patient feedback for richness and value.",
                model="qwen3:8b",
                format="plain text",
                temp=0.7,
                max_tokens=150,
            )
            # Ensure the response is stripped and converted to an integer
            score = int(response.strip())
        except Exception as e:
            # Print the error for debugging but continue processing
            print(f"\n[ERROR] Failed to score index {i}: {e}")
            score = np.nan

        # Assign score to DataFrame using the actual index 'i'
        data.at[i, 'llm_richness_score'] = score

    # Save after each batch
    data.to_csv(f'{DATA_PATH}/noema_data_scored.csv', index=False)
    # Optional: Clear memory after saving large batches
    del batch

print("✅ Done! All batches processed and saved.")
