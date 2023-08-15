import os
import json
import shutil
import subprocess
#from functools import lru_cache
from typing import List

import re
from filelock import FileLock
from rangetree import RangeTree
import numpy as np
import hashlib
from transformers import AutoTokenizer #, T5Tokenizer

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class SpecialTokenIds():
    PAD = 0
    EOS = 1
    UNK = 2
    SEP = 11
    MASK = 12

def _upper_repl(matchobj):
    #print(matchobj, matchobj.group(0))
    return '<cZ>' + matchobj.group(0).lower()

def _cap_repl(matchobj):
    #print(matchobj, matchobj.group(0))
    return matchobj.group(0)[4:].upper()

_UpperPattern = re.compile('([A-Z][a-z])')
_CapitalizedPattern = re.compile(r'(\<cZ\>[a-z])')

def pre_encode(s):
    s = _UpperPattern.sub(_upper_repl, s)
    return s.replace('\n', '<nL>').replace('\t', '<taB>')

def post_decode(s):
    return _CapitalizedPattern.sub(_cap_repl,s).replace('<nL>', '\n').replace('<taB>', '\t')

def get_tokenizer_info(tokenizer: AutoTokenizer):
    allvoc = ''.join(tokenizer.get_vocab().keys())
    sha256 = hashlib.sha256(allvoc.encode()).hexdigest()
    return dict(
        name_or_path=tokenizer.name_or_path,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        sep_token_id = tokenizer.sep_token_id,
        hash=sha256, 
        vocab_size=tokenizer.vocab_size)

def adapt_tokenizer(tokenizer: AutoTokenizer):
    if not tokenizer.sep_token_id:
        ids = tokenizer.convert_tokens_to_ids(['<sep>'])
        if ids[0] != tokenizer.unk_token_id:
            tokenizer.sep_token_id = ids[0]

    orig_tokenize = tokenizer.tokenize
    def new_tokenize(s, **kwargs):
        s=pre_encode(s)
        return orig_tokenize(s, **kwargs)
    tokenizer.tokenize = new_tokenize

    orig_encode = tokenizer.encode
    def new_encode(s, **kwargs):
        s = pre_encode(s)
        return orig_encode(s, **kwargs)
    tokenizer.encode = new_encode

    orig_encode_plus = tokenizer.encode_plus
    def new_encode_plus(s, **kwargs):
        s = pre_encode(s)
        return orig_encode_plus(s, **kwargs)
    tokenizer.encode_plus = new_encode_plus

    orig_batch_encode_plus = tokenizer.batch_encode_plus
    def new_batch_encode_plus(s, **kwargs):
        s = [pre_encode(x) for x in s]
        return orig_batch_encode_plus(s, **kwargs)
    tokenizer.batch_encode_plus = new_batch_encode_plus

    orig_convert_tokens_to_string = tokenizer.convert_tokens_to_string
    def new_convert_tokens_to_string(ss):
        s = orig_convert_tokens_to_string(ss)
        return post_decode(s)
    tokenizer.convert_tokens_to_string = new_convert_tokens_to_string

    orig_decode = tokenizer.decode
    def new_decode(args, **kwargs):
        s = orig_decode(args, **kwargs)
        return post_decode(s)
    tokenizer.decode = new_decode


def load_tokenizer(tokenizer_path="kkuramitsu/spm-pt16k"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False, trust_remote_code=True, use_fast=False)
    adapt_tokenizer(tokenizer)
    return tokenizer


DEFAULT_MAX_LENGTH = 4096*4

#@lru_cache(maxsize=63)
def load_chunks(file, remove_file=False):
    if file.endswith(".npz"):
        npz = np.load(file)
        chunk = [npz[n] for n in npz.files]
    elif not os.path.exists(file) and os.path.exists(f"{file}.zst"):
        subprocess.run(["unzstd", "--rm", f"{file}.zst"])
        with open(file, "br") as f:
            chunk = torch.load(f)
    if remove_file:
        os.remove(file)
    return chunk

class StoreDataset(Dataset):
    def __init__(self, tokenizer, cache_dir=".", select_limit=DEFAULT_MAX_LENGTH*2, **kwargs):
        self.cache_dir = cache_dir
        self.config = dict(kwargs)
        self.n_dirs = 0
        self.bufs = []
        self.n_items = 0
        self.tokenizer = tokenizer
        self.config['tokenizer'] = get_tokenizer_info(tokenizer)
        self.file_ext = self.config.get("file_ext", "npz")
        self.n_chunks = self.config.get("n_chunks", 1000)
        self.select_limit = select_limit
        self.input_lengths = []
        self.target_lengths = []

    def save_config(self):
        config_file = f"{self.cache_dir}/config.json"
        print(self.config)
        with open(config_file, "w") as w:
            json.dump(self.config, w)

    def save_chunk(self, file):
        if file.endswith('.npz'):
            np.savez_compressed(file, *self.bufs)
        elif file.endswith(".pt"):
            with open(file, "bw") as w:
                torch.save(self.bufs, w)
            subprocess.run(f"zstd -q {file} && rm {file}", shell=True)
    
    def stat_tokens(self, key, data):
        data = np.array(data)
        self.config[key] = {
            'total': int(np.sum(data)),
            'max': int(np.max(data)),
            '75%': int(np.percentile(data, 75)),
            'median': int(np.median(data)),
            '25%': int(np.percentile(data, 25)),
            'min': int(np.min(data)),
            'mean': float(np.mean(data)),
            'var': float(np.var(data)),
        }

    def save(self, save_config=True):
        if len(self.bufs) > 0:
            dir = f"{self.cache_dir}/{(self.n_dirs//100):04d}"
            os.makedirs(dir, exist_ok=True)
            filepath = f"{dir}/{self.n_dirs%100:02d}.{self.file_ext}"
            self.save_chunk(filepath)
            self.n_items += len(self.bufs)
            if len(self.bufs) == self.n_chunks:
                self.n_dirs += 1
                self.bufs = []
            else:
                self.last_chunk = filepath
        if save_config:
            self.config.update(
                dict(
                    n_items=self.n_items,
                    n_chunks=self.n_chunks,
                    n_files=self.n_dirs,
                )
            )
            if len(self.target_lengths) == 0:
                self.stat_tokens('tokens', self.input_lengths)
            else:
                self.stat_tokens('tokens', np.array(self.input_lengths)+np.array(self.target_lengths))
                self.stat_tokens('inputs', self.input_lengths)
                self.stat_tokens('labels', self.target_lengths)
            self.save_config()

    def append(self, d: List[int]):
        dtype = np.uint16 if max(d) < (2**16)-1 else np.int32
        if len(d) < self.select_limit:
            self.bufs.append(np.array(d, dtype=dtype))
        if len(self.bufs) == self.n_chunks:
            self.save(save_config=False)

    def append_text(self, text, block_size=None, max_length=DEFAULT_MAX_LENGTH):
        if isinstance(block_size, int):
            tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            block_size = block_size - self.tokenizer.num_special_tokens_to_add(pair=False)
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                # 各ブロックの先頭に特殊トークンを追加し、必要に応じてパディングを追加する
                ids = self.tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i : i + block_size]
                )
                self.append(ids)
        else:
            tokenized_text = self.tokenizer.encode(text, truncation=True, max_length=max_length)
            if len(tokenized_text) < self.select_limit:
                self.input_lengths.append(len(tokenized_text))
                self.append(tokenized_text)

    def append_pair(self, text, text2, max_length=DEFAULT_MAX_LENGTH, target_max_length=None):
        assert self.tokenizer.sep_token_id is not None
        target_max_length = target_max_length or max_length
        text = self.tokenizer.encode(text, truncation=True, max_length=max_length)
        text[-1] = self.tokenizer.sep_token_id
        text2 = self.tokenizer.encode(text2, truncation=True, max_length=target_max_length)
        if len(text)+len(text2) < self.select_limit:
            self.input_lengths.append(len(text))
            self.target_lengths.append(len(text2))
            self.append(text+text2)
    
    def chunkpath(self, i):
        n = i // self.n_chunks
        return f"{(n//100):04d}/{n%100:02d}.{self.file_ext}"

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        i = i % self.n_items
        chunkpath = self.chunkpath(i)
        filepath = f"{self.cache_dir}/{chunkpath}"
        chunks = load_chunks(filepath, remove_file=False)
        return chunks[i % self.n_chunks]

#    cmd = "aria2c -x5 -o {0} {1}".format(url.split('/')[-1], url)
#    subprocess.call(cmd, shell=True)


def download(url, dir, chunkpath, zext="", sync=True):
    remote_file = f"{url}/{chunkpath}{zext}"
    local_file = f"{dir}/{chunkpath}{zext}"
    local_dir, _, _ = local_file.rpartition("/")
    os.makedirs(local_dir, exist_ok=True)
    print('downloading', remote_file)
    if remote_file.startswith('file:'):
        remote_file = os.path.abspath(remote_file[5:]) # file: をとる
        subprocess.call(f'cp {remote_file} {local_file}', shell=True)
        return
    cmd = f"wget -qO {local_file} {remote_file}"
    if not sync:
        cmd = f"{cmd} &"
    subprocess.call(cmd, shell=True)

from collections import deque

ID = 0
def random_name():
    global ID
    ID+= 1
    return f'A{ID-1}'

class ChunkedDataset(Dataset):
    def __init__(self, url, lock_file='lock', cache_dir=".", **kwargs):
        self.cache_dir = f'{cache_dir}/{random_name()}'
        self.url = url
        self.lock_file = lock_file
        self.config = self.load_config(kwargs)
        self.file_ext = self.config.get("file_ext", "npz")
        self.n_items = self.config.get("n_items", 0)
        self.n_chunks = self.config.get("n_chunks", 1000)
        self.queue = deque(maxlen=64)
        self.cache = {}

    def load_config(self, kwargs: dict):
        with FileLock(self.lock_file):
            os.makedirs(self.cache_dir, exist_ok=True)
            config_file = f"{self.cache_dir}/config.json"
            if self.url and not os.path.exists(config_file):
                download(self.url, self.cache_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = dict(n_items=0)
        config.update(kwargs)
        return config
    
    def __len__(self):
        return self.n_items

    def get_chunks(self, chunkpath):
        if chunkpath in self.cache:
            return self.cache[chunkpath]
        filepath = f"{self.cache_dir}/{chunkpath}"
        with FileLock(self.lock_file):
            if not os.path.exists(filepath):
                download(self.url, self.cache_dir, chunkpath)
            # prefetch = self.chunkpath((i + self.n_chunks) % self.n_items)
            # if not os.path.exists(f"{self.cache_dir}/{prefetch}"):
            #     download(self.url, self.cache_dir, prefetch, sync=False)
        with FileLock(self.lock_file):
            chunks = load_chunks(filepath)
        if len(self.queue) == 64:
            older = self.queue[0]
            if older in self.cache:
                del self.cache[older]
            try:
                with FileLock(self.lock_file):
                    os.remove(f'{self.cache_dir}/{older}')
            except FileNotFoundError:
                pass
        self.queue.append(chunkpath)
        self.cache[chunkpath] = chunks
        return chunks

    def chunkpath(self, i):
        n = i // self.n_chunks
        return f"{(n//100):04d}/{n%100:02d}.{self.file_ext}"

    def __getitem__(self, i):
        i = i % self.n_items
        chunkpath = self.chunkpath(i)
        chunks = self.get_chunks(chunkpath)
        return chunks[i % self.n_chunks]

    def reset(self):
        self.cache = {}
        for chunkpath in self.deque:
            os.remove(f'{self.cache_dir}/{chunkpath}')
        self.deque = deque(maxlen=64)

def build_inputs_for_clm(data, max_length=None):
    if isinstance(max_length, int) and len(data) > max_length:
        data[max_length-1]=data[-1]
        return torch.tensor(data[:max_length].astype(np.int32), dtype=torch.long)
    else:
        return torch.tensor(data.astype(np.int32), dtype=torch.long)

import random

def build_inputs_for_seq2seq(data, max_length=None, target_max_length=None):
    target_max_length = target_max_length or max_length
    eos_id = data[-1]
    index = data.tolist().index(SpecialTokenIds.SEP)
    if index > 0:
        inputs = data[:index+1]
        labels = data[index+1:]
    else:
        inputs = []
        for x in data[:-1]:
            if random.random() < 0.3:
                if len(inputs) > 0 and inputs[-1] != SpecialTokenIds.MASK:
                    inputs.append(SpecialTokenIds.MASK)
                continue
            inputs.append(x)
        inputs.append(eos_id)
        inputs=np.array(inputs)
        labels=data
    if isinstance(max_length, int) and len(inputs) > max_length:
        inputs = inputs[:max_length]
        inputs[max_length-1]=eos_id
    if isinstance(target_max_length, int) and len(labels) > target_max_length:
        labels = labels[:target_max_length]
        labels[target_max_length-1]=eos_id
    return {
        "input_ids": torch.tensor(inputs.astype(np.int32), dtype=torch.long),
        # "attention_mask": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels.astype(np.int32), dtype=torch.long),
    }

# url
# https://papertown/papertown/nlp

def _load_compose(c, urls):
    if isinstance(urls, str):
        urls = urls.split('|')
    for url in urls:
        if isinstance(url, str):
            if url.find(':', 8) > 8:
                url, _, split = url.rpartition(':')
                split = float(split)
                if split > 1.0:
                    split = int(split)
                c.add(url, split)
            else:
                c.add(url)
        elif isinstance(url, dict):
            if 'split' in url:
                c.add(url['url'], url['split'])
            else:
                c.add(url['url'])

class DataComposer(Dataset):
    def __init__(self, urls=[], cache_dir = '.', indexer=None, seq2seq=False, max_length=None):
        self.cache_dir = f'{cache_dir}/cache{random_name()}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock_file = f'{self.cache_dir}/lock'
        self.max_length=max_length
        self.n_items = 0
        self.indexer = indexer
        self.rangetree = RangeTree()
        _load_compose(self, urls)
        if seq2seq:
            self.build_fn = build_inputs_for_seq2seq
        else:
            self.build_fn = build_inputs_for_clm

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.rangetree = None
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def add(self, url:str, split=1):
        offset = self.n_items
        chunk = ChunkedDataset(url, self.lock_file, cache_dir=f'{self.cache_dir}/{offset}')
        n_items = len(chunk)
        if isinstance(split, float) and split < 1:
            n_items = int(n_items * split)
        elif split != 1:
            n_items = min(n_items, int(split))
        self.n_items += n_items
        self.rangetree[offset:self.n_items] = (chunk, offset)

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        if self.indexer is not None:
            idx = self.indexer % self.n_items
            self.indexer +=1
        chunk, offset = self.rangetree[idx]
        return self.build_fn(chunk[idx-offset], self.max_length)

    def set_build(self, build_fn):
        self.build_fn = build_fn
