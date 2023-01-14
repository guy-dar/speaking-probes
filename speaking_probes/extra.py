from copy import deepcopy
import heapq
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# projection to embedding space
ALNUM_CHARSET = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

def convert_to_tokens(indices, tokenizer, strip=True):
    res = tokenizer.convert_ids_to_tokens(indices)
    if strip:
        res = list(map(lambda x: x[1:] if x[0] == 'Ġ' else "#" + x, res))
    return res


def top_tokens(v, k=20, tokenizer=None, only_alnum=False, only_ascii=True):
    v = deepcopy(v)
    ignored_indices = []
    if only_ascii:
        ignored_indices.extend([key for val, key in tokenizer.vocab.items() if not val.strip('Ġ▁').isascii()])
    if only_alnum: 
        ignored_indices.extend([key for val, key in tokenizer.vocab.items() if not (set(val.strip('Ġ▁[] ')) <= ALNUM_CHARSET)])
    ignored_indices = list(set(ignored_indices))
    v[ignored_indices] = -np.inf
    values, indices = torch.topk(v, k=k)
    return convert_to_tokens(indices, tokenizer, strip=True)


def to_embedding_space(neuron, emb, tokenizer, k=10):
    return top_tokens(neuron @ emb, tokenizer=tokenizer, k=k)


# search in corpus
def dict_to_device(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

class TopkTracker:
    def __init__(self, k):
        self.heap = []
        self.k = k
        
    def add(self, key, score):
        x = TopkTracker._Item(key, score)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, x)
        else:
            heapq.heappushpop(self.heap, x)

    def extend(self, keys, scores):
        for key, score in zip(keys, scores):
            self.add(key, score)
    
    def __iter__(self):
        return iter(sorted(self.heap, key=lambda item: item.score, reverse=True))
    
    def keys(self):
        return list(map(lambda x: x.key, iter(self)))
    
    class _Item:
        def __init__(self, key, score):
            self.key = key
            self.score = score
        
        def __lt__(self, other):
            return self.score < other.score
        
        def __repr__(self):
            return f"({self.key},{self.score})"
        

def corpus_search(model, tokenizer, corpus, layer_path, dim_idx, k=5,
                 max_seq_length=128, batch_size=4, normalize=False):    
    def _hook_fn(self, inp, outp):
        outp = outp.detach().cpu()
        if normalize:
            outp = F.normalize(outp, dim=-1)
        activation = outp[:, :, dim_idx]
        max_acts, max_idxs = activation.max(dim=1)
        for i in range(len(activation)):
            topk_tracker.add(key=(sentences[i], max_idxs[i].item()), score=max_acts[i].item())
    
    topk_tracker = TopkTracker(k=k)
    dataloader = DataLoader(corpus, batch_size=batch_size)
    hook = model.get_submodule(layer_path).register_forward_hook(_hook_fn)
    try:
        for sentences in tqdm(dataloader):
            inputs = tokenizer(sentences, padding=True, truncation=True, 
                               max_length=max_seq_length, return_tensors='pt')
            model(**dict_to_device(inputs, model.device))
        hook.remove()
    except e:
        print(e)
    finally:
        print("removing hook..")
        hook.remove()
        return topk_tracker.keys()