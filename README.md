# Noema 🧠

**Noema** is a Feedback Intelligence Dashboard designed to analyze, score, and interpret free-text feedback. It combines natural language processing, machine learning, and large language model (LLM) insights to provide actionable understanding of feedback trends, richness, and sentiment over time.

---

## 🔹 Features

- **Hybrid Richness Scoring**
  - Linguistic, semantic, and LLM-based scoring of textual feedback.
  - Generates a 0–100 richness score for each feedback item.

- **Sentiment Analysis**
  - Measures polarity and subjectivity of feedback.
  - Visualizes sentiment trends alongside richness.

- **Topic Clustering**
  - Groups feedback into themes using BERTopic.
  - Identify recurring concerns and praise.

- **Temporal Tracking**
  - Tracks richness and sentiment trends over time (weekly/monthly).
  - Rolling averages smooth short-term fluctuations.

- **Event Analysis**
  - **Manual intervention detection**: Compare pre/post metrics around known events using t-tests.
  - **Automatic change-point detection**: Finds structural shifts in richness or sentiment without user input.

- **LLM Explanations**
  - Automatically generates concise, human-readable summaries describing what changed around detected events.

- **Interactive Dashboard**
  - Built with Streamlit.
  - Dynamic plots with Plotly: richness distribution, sentiment vs richness, topic frequency, trend charts with annotated events.
  - CSV export of full analysis.

---
