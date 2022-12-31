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
    

col1, col2, col3, *_ = st.columns(5)
model_name = col1.selectbox("Select a model: ", options=['gpt2', 'gpt2-medium', 'gpt2-large'])
model, model_params, tokenizer = load_model(model_name)
neuron_layer = col2.text_input("Layer: ", value='0')
neuron_dim = col3.text_input("Dim: ", value='0')

neurons = model_params.K_heads[int(neuron_layer), int(neuron_dim)]
prompt = st.text_area("Prompt: ")
submitted = st.button("Send!")

if submitted:
    with st.spinner('Wait for it..'):
        model, model_params, tokenizer = map(deepcopy, (model, model_params, tokenizer))
        decoded = speaking_probe(model, model_params, tokenizer, prompt, *neurons, 
                                 repetition_penalty=2., num_generations=3,
                                 min_length=1, do_sample=True, 
                                 max_new_tokens=100)

    for text in decoded:
        st.text(textwrap.wrap(text, width=70))
