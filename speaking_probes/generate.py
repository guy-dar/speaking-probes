import re
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from torch import nn
import torch 
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel
import gc
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from torch import nn
import torch 
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, MaxLengthCriteria, StoppingCriteriaList
from transformers import DataCollatorWithPadding
from transformers import LogitsProcessor, LogitsProcessorList, LogitsWarper
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
from dataclasses import dataclass
from argparse import ArgumentParser


@dataclass
class ModelParameters:
    K_heads: torch.Tensor
    num_layers: int
    d_int: int
    num_heads: int
    hidden_dim: int
    head_size: int
    V_heads: torch.Tensor = None
    W_Q_heads: torch.Tensor = None
    W_K_heads: torch.Tensor = None
    W_V_heads: torch.Tensor = None
    W_O_heads: torch.Tensor = None
    emb: torch.Tensor = None
    
    

def extract_gpt_parameters(model, full=False):
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    hidden_dim = model.config.n_embd
    head_size = hidden_dim // num_heads

    K = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_fc.weight").T
                               for j in range(num_layers)]).detach()

    K_heads = K.reshape(num_layers, -1, hidden_dim)
    d_int = K_heads.shape[1]
    model_params = ModelParameters(K_heads=K_heads, num_layers=num_layers, d_int=d_int,
                                   hidden_dim=hidden_dim, head_size=head_size, 
                                   num_heads=num_heads)

    if full:
        emb = model.get_output_embeddings().weight.data.T
        V = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
                               for j in range(num_layers)]).detach()
        W_Q, W_K, W_V = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight") 
                                   for j in range(num_layers)]).detach().chunk(3, dim=-1)
        W_O = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight") 
                                   for j in range(num_layers)]).detach()

        model_params.V_heads = V.reshape(num_layers, -1, hidden_dim)
        model_params.W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
        model_params.W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)
        model_params.W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
        model_params.W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
        model_params.emb = emb
    return model_params


def extract_gpt_j_parameters(model, full=False):
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    hidden_dim = model.config.n_embd
    head_size = hidden_dim // num_heads

    K = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.fc_in.weight").T
                               for j in range(num_layers)]).detach()

    K_heads = K.reshape(num_layers, -1, hidden_dim)
    d_int = K_heads.shape[1]
    model_params = ModelParameters(K_heads=K_heads, num_layers=num_layers, d_int=d_int,
                                   hidden_dim=hidden_dim, head_size=head_size, 
                                   num_heads=num_heads)

    if full:
        raise NotImplementedError
#         emb = model.get_output_embeddings().weight.data.T
#         V = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
#                                for j in range(num_layers)]).detach()
#         W_Q, W_K, W_V = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight") 
#                                    for j in range(num_layers)]).detach().chunk(3, dim=-1)
#         W_O = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight") 
#                                    for j in range(num_layers)]).detach()

#         model_params.V_heads = V.reshape(num_layers, -1, hidden_dim)
#         model_params.W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
#         model_params.W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)
#         model_params.W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
#         model_params.W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
#         model_params.emb = emb
    return model_params


def encode(token, tokenizer):
    assert (type(token) == str)
    encoded = tokenizer.encode(token)
    assert (len(encoded) == 1)
    return encoded[0]


def read_and_go(path):
    with open(path, 'r') as f:
        return f.read()
    

def extend_model_and_tokenizer(model, model_params, tokenizer, min_layer=0, 
                               max_layer=None):
    if max_layer is None:
        max_layer = len(model_params.K_heads)-1
    relevant_neurons = model_params.K_heads[min_layer:max_layer+1]
    num_regular_tokens = len(tokenizer)
    new_tokens = [f" <param_{layer}_{dim}>"  for layer in range(min_layer, max_layer+1) 
                                             for dim in range(relevant_neurons.shape[1])]

    tokenizer_extended = deepcopy(tokenizer)
    model_extended = deepcopy(model)

    tokenizer_extended.add_tokens(new_tokens)
    model_extended.resize_token_embeddings(len(tokenizer_extended))
    model_extended.transformer.wte.weight.data[-len(new_tokens):] = relevant_neurons.flatten(0, -2)
    return model_extended, tokenizer_extended
    
    
# logit processors
class NeuronTokenBan(LogitsWarper):
    def __init__(self, num_non_neuron_tokens, ban_penalty=-np.inf):
        self.ban_penalty = ban_penalty
        self.num_non_neuron_tokens = num_non_neuron_tokens
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        scores[:, self.num_non_neuron_tokens:] = self.ban_penalty
        return scores
    
    
class ParamListStructureEnforcer(LogitsProcessor):
    def __init__(self, tokenizer, num_regular_tokens):
        self.tokenizer = tokenizer
        self.num_regular_tokens = num_regular_tokens
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        last_input_id = input_ids[0, -1]
        tokenizer = self.tokenizer
        num_regular_tokens = self.num_regular_tokens
        
        comma_id = encode(',', tokenizer)
        eos_score, comma_score = deepcopy(scores[:, tokenizer.eos_token_id]), deepcopy(scores[:, comma_id])
        
        if last_input_id >= num_regular_tokens:
            scores[:] = -np.inf
            scores[:, comma_id] = comma_score
        else:
            scores[:, :num_regular_tokens] = -np.inf           
        
        scores[:, tokenizer.eos_token_id] = eos_score
        return scores
    

# speaking probe
def _preprocess_prompt(model_params, prompt):
    K_heads = model_params.K_heads
    prompt = re.sub(r'([^ ]|\A)(<neuron>|<param_\d+_\d+>)', lambda m: f'{m.group(1)} {m.group(2)}', prompt)
    param_neuron_idxs = [(int(a), int(b)) for a, b in re.findall(r' <param_(\d+)_(\d+)>', prompt)]
    param_neuron_tokens = [f' <param_{a}_{b}>' for a, b in param_neuron_idxs]
    param_neurons = [deepcopy(K_heads[a, b]) for a, b in param_neuron_idxs]
    return prompt, param_neuron_tokens, param_neurons


def speaking_probe(model, model_params, tokenizer, prompt, *neurons,
                   num_generations=1, layer_range=None, bad_words_ids=[], output_neurons=False,
                   return_outputs=False, logits_processor=LogitsProcessorList([]), **kwargs):
    num_non_neuron_tokens = len(tokenizer)
    tokenizer_with_neurons = deepcopy(tokenizer)
    
    # adding neurons to the tokenizer
    neuron_tokens = [f" <neuron{i+1 if i > 0 else ''}>" for i in range(len(neurons))]
    prompt, param_neuron_tokens, param_neurons = _preprocess_prompt(model_params, prompt)
    neuron_tokens.extend(param_neuron_tokens)
    neurons = neurons + tuple(param_neurons)
    has_extra_neurons = len(neurons) > 0
    if has_extra_neurons:
        tokenizer_with_neurons.add_tokens(neuron_tokens)
        model.resize_token_embeddings(len(tokenizer_with_neurons))
        model.transformer.wte.weight.data[-len(neurons):] = torch.stack(neurons, dim=0)
        
    logits_processor = deepcopy(logits_processor)
    
    if not output_neurons:
        logits_processor.append(NeuronTokenBan(num_non_neuron_tokens))
    
    if layer_range is not None:
        num_layers = model_params.num_layers
        min_layer, max_layer = layer_range
        bad_words_ids = deepcopy(bad_words_ids)
        bad_words_ids.extend([[encode(f" <param_{i}_{j}>", tokenizer)] 
                                        for j in range(model_params.d_int) 
                                        for i in [*range(min_layer), *range(max_layer+1, num_layers)]])
    if len(bad_words_ids) == 0:
        bad_words_ids = None
        
    input_ids = tokenizer_with_neurons.encode(prompt, return_tensors='pt').to(model.device)
    input_ids = torch.cat([deepcopy(input_ids) for _ in range(num_generations)], dim=0)
    outputs = model.generate(input_ids, pad_token_id=model.config.eos_token_id, 
                             logits_processor=logits_processor, 
                             bad_words_ids=bad_words_ids,
                             return_dict_in_generate=True,
                             **kwargs)

    decoded = tokenizer_with_neurons.batch_decode(outputs.sequences, skip_special_tokens=True)
    
    # TODO: add `finally` statement
    if has_extra_neurons:
        model.resize_token_embeddings(num_non_neuron_tokens)
        model.transformer.wte.weight.data = model.transformer.wte.weight.data[:num_non_neuron_tokens]
        
    if return_outputs:
        return decoded, outputs
    else:
        return decoded


# main
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, default=None)
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--neuron', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--no_sample', action='store_true')
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--num_generations', type=int, default=1)
    parser.add_argument('--min_length', type=int, default=20)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--repetition_penalty', type=float, default=2.)
    parser.add_argument('--temperature', type=float, default=1.)
    
    args = parser.parse_args()
    # TODO: first make them mutually exclusive 
    if args.max_new_tokens is not None:
        args.max_length = None
    
    
    print("loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model_params = extract_gpt_parameters(model)
    prompt = args.prompt or read_and_go(args.prompt_file)
    device = args.device
    model = model.to(device)
    
    i1, i2 = map(lambda x: int(x.strip()), args.neuron.split(','))
    neuron = model_params.K_heads[i1, i2]
    neurons = [neuron]

    print(prompt)
    decoded = speaking_probe(model, model_params, tokenizer, prompt, *neurons,
                   num_generations=args.num_generations, 
                   repetition_penalty=args.repetition_penalty, 
                   num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                   temperature=args.temperature,
                   min_length=args.min_length, do_sample=not args.no_sample,
                   max_length=args.max_length, max_new_tokens=args.max_new_tokens)
    for i in range(len(decoded)):
        print("\n\ngenerate:", decoded[i])
