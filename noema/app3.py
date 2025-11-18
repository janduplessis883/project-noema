import streamlit as st

# Initialize the toggle state (optional, but good practice)
if 'model_toggle' not in st.session_state:
    st.session_state.model_toggle = False # Off by default

# Set the toggle value from session state
is_ollama_on = st.session_state.model_toggle

# Dynamically construct the label based on the current state
if is_ollama_on:
    # ON mode: Ollama is bold
    label_text = "OpenAI | **Ollama**"
else:
    # OFF mode: OpenAI is bold
    label_text = "**OpenAI** | Ollama"

# Display the toggle
# We use the key to link the widget value to session state
st.toggle(
    label=label_text,
    value=is_ollama_on,
    key='model_toggle'
)

# Optional: Display the selected model
selected_model = "Ollama" if is_ollama_on else "OpenAI"
st.write(f"The selected model is: **{selected_model}**")
