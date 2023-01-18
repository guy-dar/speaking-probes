import streamlit as st
from speaking_probes.generate import extract_gpt_parameters, speaking_probe
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
import textwrap


@st.cache
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_params = extract_gpt_parameters(model)
    return model, model_params, tokenizer
    

col1, col2, col3, *_ = st.columns(4)
model_name = col1.selectbox("Select a model: ", options=['gpt2', 'gpt2-medium', 'gpt2-large'])
model, model_params, tokenizer = load_model(model_name)
# neuron_layer = col2.text_input("Layer: ", value='0')
# neuron_dim = col3.text_input("Dim: ", value='0')
# neurons = model_params.K_heads[int(neuron_layer), int(neuron_dim)]

with st.sidebar:
    temperature = st.slider("Temperature", min_value=0., max_value=2., value=0.5, step=0.05)
    repetition_penalty = st.slider("Repetition Penalty", min_value=0., max_value=4., value=2., step=0.1)
    sidebar_cols = st.columns(2)
    num_generations = sidebar_cols[0].number_input("Number of Answers", min_value=1, value=3, format='%d')
    max_new_tokens = sidebar_cols[1].number_input("Max Answer Length", min_value=1, value=50, format='%d')

prompt = st.text_area("Prompt: ")
submitted = st.button("Send!")

if submitted:
    with st.spinner('Wait for it..'):
        model, model_params, tokenizer = map(deepcopy, (model, model_params, tokenizer))
        decoded = speaking_probe(model, model_params, tokenizer, prompt, 
                                 repetition_penalty=repetition_penalty, num_generations=num_generations,
                                 min_length=1, do_sample=True, 
                                 max_new_tokens=max_new_tokens, temperature=temperature)

    for text in decoded:
        st.code('\n'.join(textwrap.wrap(text, width=70)), language=None)
