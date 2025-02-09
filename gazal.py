import streamlit as st
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

model = load_model('ghazal_generator_best.keras')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_seq_length = 30

def generate_ghazal(seed_text, num_lines=8, temperature=1.0):
    generated_text = ""
    first_line = True

    for _ in range(num_lines):
        if first_line:
            line = seed_text
            first_line = False
        else:
            line = ""

        for _ in range(8):
            tokenized_input = tokenizer.texts_to_sequences([line.strip()])[0]
            tokenized_input = pad_sequences([tokenized_input], maxlen=max_seq_length-1, padding='pre')

            predictions = model.predict(tokenized_input)[0]

            predictions = np.log(predictions + 1e-8) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)

            predicted_index = np.random.choice(len(predictions), p=predictions)
            next_word = tokenizer.index_word.get(predicted_index, "")

            if not next_word:
                break

            line += " " + next_word

        if random.random() < 0.5 and not first_line:
            words_in_line = line.split()
            insert_pos = random.randint(1, len(words_in_line))
            words_in_line.insert(insert_pos, seed_text)
            line = " ".join(words_in_line)

        generated_text += "\n" + line.strip()

    return generated_text.strip()

st.set_page_config(page_title="Gehri Baten", page_icon=":sparkles:", layout="centered")

st.markdown("""
    <h1 style="text-align: center; color: #FF7043;">Gehri Baten</h1>
    <h3 style="text-align: center; color: #4FC3F7;">A Poetry Generator in Roman Urdu</h3>
    <p style="text-align: center; font-size: 16px; color: #9E9E9E;">Enter a seed word or sentence to generate a unique ghazal.</p>
""", unsafe_allow_html=True)

seed_text = st.text_input("Seed Text", "sanson ki mala")
num_lines = st.slider("Number of Lines", min_value=1, max_value=15, value=8)
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.button("Generate Ghazal", use_container_width=True):
    if seed_text:
        st.write("**Generated Ghazal:**")
        generated_ghazal = generate_ghazal(seed_text, num_lines=num_lines, temperature=temperature)
        st.text(generated_ghazal)
    else:
        st.write("Please enter a seed text.")
