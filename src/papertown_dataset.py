import os
import time
import random

import json
import shutil
import subprocess
from typing import List

import re
from collections import deque

from filelock import FileLock
import numpy as np
import hashlib

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

def verbose_print(*args, **kwargs):
    print('🐜', *args, **kwargs)


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
        vocab_domain = tokenizer.name_or_path.replace('/', '_'),
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        sep_token_id = tokenizer.sep_token_id,
        newline_token_id = tokenizer.newline_token_id,
        hash=sha256, 
        vocab_size=tokenizer.vocab_size)

def adapt_tokenizer(tokenizer: AutoTokenizer):
    if not tokenizer.sep_token_id:
        ids = tokenizer.convert_tokens_to_ids(['<sep>'])
        if ids[0] != tokenizer.unk_token_id:
            tokenizer.sep_token_id = ids[0]
    ids = tokenizer.convert_tokens_to_ids(['<nL>'])
    if ids[0] != tokenizer.unk_token_id:
        tokenizer.newline_token_id = ids[0]

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

def load_tokenizer(tokenizer_path="kkuramitsu/spm-pt16k", adapt=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False, trust_remote_code=True, use_fast=False)
    if adapt:
        adapt_tokenizer(tokenizer)
    return tokenizer

import gzip


DEFAULT_MAX_LENGTH = 4096 * 4
MAX_SELECT_LIMIT = DEFAULT_MAX_LENGTH*2
N_CHUNKS = 4096

def chunk_filename(dir, chunkseq, prefix, file_ext, mkdir=True):
    dir = f"{dir}/{(chunkseq//100):04d}"
    if mkdir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return f"{dir}/{prefix}{(chunkseq%100):02d}.{file_ext}"

def save_chunk(dir, chunkseq, prefix, file_ext, chunks):
    filename = chunk_filename(dir, chunkseq, prefix, file_ext)
    if filename.endswith('.npz'):
        np.savez_compressed(filename, *chunks)
    elif filename.endswith('.jsonl.gz'):
        with gzip.open(filename, 'wt') as w:
            for chunk in chunks:
                print(json.dumps(chunk, ensure_ascii=False), file=w)

def zopen(filepath):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    else:
        return open(filepath, 'r')

def load_chunk_npz(filename):
    npz = np.load(filename)
    # return [npz[n].astype(np.int32) for n in npz.files]
    return [npz[n] for n in npz.files]


def load_chunk(dir, chunkseq, prefix, file_ext):
    filename = chunk_filename(dir, chunkseq, prefix, file_ext, mkdir=False)
    if filename.endswith(".npz"):
        chunks = load_chunk_npz(filename)
    elif filename.endswith("jsonl") or filename.endswith('jsonl.gz'):
        chunks=[]
        with zopen(filename) as f:
            for line in f.readlines():
                d = json.loads(line)
                chunks.append(d)
    else:
        chunks=[]
        with zopen(filename) as f:
            for line in f.readlines():
                chunks.append(line.strip().replace('<nl>', '<nL>'))
    return chunks

def map_chunk(dir, chunkseq, prefix, map_fn):
    chunks = load_chunk(dir, chunkseq, '', 'jsonl.gz')
    chunks = [map_fn(chunk) for chunk in chunks]
    save_chunk(dir, chunkseq, prefix, prefix, 'npz', chunks)
    return [len(chunk) for chunk in chunks]

def _block_add(tokenizer, blocks, tokens, block_size):
    for i in range(0, len(tokens) - block_size + 1, block_size):  
        segmented = tokens[i : i + block_size]
        ids = tokenizer.build_inputs_with_special_tokens(segmented)
        if len(ids) > 0:
            blocks.append(ids)


def tokenize_block(tokenizer, text, block_size=256, newline_token_id = None):
    blocks = []
    block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
    text = text.replace('\r\n', '<nL>').replace('\n', '<nL>')
    if newline_token_id:
        if '<nL><nL>' in text:
            lines = text.split('<nL><nL>')
            NL = [newline_token_id, newline_token_id]
        else:
            lines = text.split('<nL>')
            NL = [newline_token_id]

        buffer = []
        for line in lines:
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))+NL
            if len(buffer) + len(tokens) > block_size:
                if len(buffer) < (block_size * 4 // 5):  # 80% 埋まってなければオーバラップ
                    buffer.extend(tokens) # overlaping
                ids = tokenizer.build_inputs_with_special_tokens(buffer[:block_size])
                blocks.append(ids)
                buffer=tokens
                continue
            buffer.extend(tokens)
        if len(buffer) > (block_size // 2):
            _block_add(tokenizer, blocks, buffer, block_size)
        else:
            buffer = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lines[-1]))
            for line in lines[:-1][::-1]:
                tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))+NL
                if len(buffer) + len(tokens) > block_size:
                    break
                buffer = tokens + buffer
            _block_add(tokenizer, blocks, buffer, block_size)
    else:
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        _block_add(tokenizer, blocks, tokens, block_size)
    return blocks


def tokenize_text(tokenizer, text, max_length=DEFAULT_MAX_LENGTH):
    return tokenizer.encode(text, truncation=True, max_length=max_length)

def tokenize_pair(tokenizer, text, text2, max_length=DEFAULT_MAX_LENGTH, target_max_length=None):
    assert tokenizer.sep_token_id is not None
    target_max_length = target_max_length or max_length
    text = tokenizer.encode(text, truncation=True, max_length=max_length)
    text[-1] = tokenizer.sep_token_id
    text2 = tokenizer.encode(text2, truncation=True, max_length=target_max_length)
    return text+text2

def stat_tokens(counts):
    if len(counts) == 0:
        return {'total': 0}
    data = np.array(counts)
    return {
        'total': int(np.sum(data)),
        'max': int(np.max(data)),
        '75%': int(np.percentile(data, 75)),
        'median': int(np.median(data)),
        '25%': int(np.percentile(data, 25)),
        'min': int(np.min(data)),
        'mean': float(np.mean(data)),
        'std': float(np.var(data)) ** 0.5,
    }

class DatasetStore(object):
    def __init__(self, cache_dir=".", select_limit=DEFAULT_MAX_LENGTH*2, **kwargs):
        self.cache_dir = cache_dir
        self.config = dict(kwargs)
        self.tokenizer = self.config.get("tokenizer", None)
        if self.tokenizer:
            info = get_tokenizer_info(self.tokenizer)
            self.config['tokenizer'] = info
            self.vocab_domain = self.config.get("vocab_domain", info['vocab_domain'])
            self.file_ext = self.config.get("file_ext", "npz")
        else:
            self.vocab_domain = ''
            self.file_ext = self.config.get("file_ext", "jsonl.gz")
        self.n_chunks = self.config.get("n_chunks", N_CHUNKS)
        self.chunkseq = 0
        self.bufs = []
        self.n_items = 0
        self.select_limit = select_limit
        self.config['select_limit'] = select_limit
        self.token_counts = []

    def save_config(self):
        config_file = f"{self.cache_dir}/{self.vocab_domain}config.json"
        with open(config_file, "w") as w:
            json.dump(self.config, w)
    
    def save(self, save_config=True):
        if len(self.bufs) > 0:
            save_chunk(self.cache_dir, self.chunkseq, self.vocab_domain, self.file_ext, self.bufs)
            self.n_items += len(self.bufs)
            if len(self.bufs) == self.n_chunks:
                self.chunkseq += 1
                self.bufs = []
        if save_config:
            self.config.update(
                dict(
                    n_items=self.n_items,
                    n_chunks=self.n_chunks,
                    chunkseq=self.chunkseq,
                    tokens=stat_tokens(self.token_counts)
                )
            )
            self.save_config()

    def append(self, d: List[int]):
        if len(d) < self.select_limit:
            self.bufs.append(np.array(d, dtype=np.int32))
            self.token_counts.append(len(d))
        if len(self.bufs) == self.n_chunks:
            self.save(save_config=False)

    def append_dict(self, d: List[int]):
        self.bufs.append(d)
        if len(self.bufs) == self.n_chunks:
            self.save(save_config=False)

    def append_block(self, text, block_size=256):
        newline_token_id = None
        if hasattr(self.tokenizer, 'newline_token_id'):
            newline_token_id = self.tokenizer.newline_token_id
        blocks = tokenize_block(self.tokenizer, text, block_size, newline_token_id)
        for b in blocks:
            self.append(b)

    def append_text(self, text, block_size=None, max_length=DEFAULT_MAX_LENGTH):
        if isinstance(block_size, int):
            self.append_block(text, block_size=block_size)
        else:
            self.append(tokenize_text(self.tokenizer, text, max_length=max_length))

    def append_pair(self, text, text2, max_length=DEFAULT_MAX_LENGTH, target_max_length=None):
        self.append(tokenize_pair(self.tokenizer, text, text2, max_length=max_length))

    def __len__(self):
        # 切り上げ
        return (self.n_items + self.n_chunks - 1) // self.n_chunks

    def __getitem__(self, i):
        return chunk_filename(self.cache_dir, i, self.vocab_domain, self.file_ext, mkdir=False)

#    cmd = "aria2c -x5 -o {0} {1}".format(url.split('/')[-1], url)
#    subprocess.call(cmd, shell=True)

def download(url, dir, local_file, sync=True):
    file=local_file[len(dir):]
    remote_file = f"{url}/{file}"
    local_dir, _, _ = local_file.rpartition("/")
    os.makedirs(local_dir, exist_ok=True)
    if remote_file.startswith('file:'):
        remote_file = os.path.abspath(remote_file[5:]) # file: をとる
        cmd = f'cp {remote_file} {local_file}'
    else:
        cmd = f"wget -qO {local_file} {remote_file}"
    if sync:
        verbose_print('downloading', cmd)
    else:
        cmd = f"{cmd} &"
    subprocess.call(cmd, shell=True)


ID = 0
def random_name():
    global ID
    ID+= 1
    return f'Cache{ID-1}'

class ChunkedDataset(Dataset):
    def __init__(self, url, vocab_domain, lock_file='lock', cache_dir=".", prefetch=3, **kwargs):
        self.url = url
        self.vocab_domain = vocab_domain
        self.cache_dir = f'{cache_dir}/{random_name()}'
        self.lock_file = lock_file
        self.config = self.load_config(kwargs)
        self.file_ext = self.config.get("file_ext", "npz")
        self.n_items = self.config.get("n_items", 0)
        self.n_chunks = self.config.get("n_chunks", N_CHUNKS)
        self.prefetch=prefetch
        self.queue = deque(maxlen=64)
        self.cache = {}

    def load_config(self, kwargs: dict):
        with FileLock(self.lock_file):
            os.makedirs(self.cache_dir, exist_ok=True)
            config_file = f"{self.cache_dir}/{self.vocab_domain}config.json"
            if self.url and not os.path.exists(config_file):
                download(self.url, self.cache_dir, config_file)
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = dict(n_items=0)
        config.update(kwargs)
        return config
    
    def __len__(self):
        return self.n_items

    def get_chunks(self, filepath):
        if filepath in self.cache:
            return self.cache[filepath]
        with FileLock(self.lock_file):
            if not os.path.exists(filepath):
                download(self.url, self.cache_dir, filepath)
        with FileLock(self.lock_file):
            chunks = load_chunk_npz(filepath)
            random.shuffle(chunks)
        if len(self.queue) == 64:
            older = self.queue.popleft()
            if older in self.cache:
                del self.cache[older]
            try:
                os.remove(f'{self.cache_dir}/{older}')
            except FileNotFoundError:
                pass
        self.queue.append(filepath)
        self.cache[filepath] = chunks
        return chunks

    def __getitem__(self, i):
        #i = i % self.n_items
        chunkseq = i // self.n_chunks
        filepath = chunk_filename(self.cache_dir, chunkseq, self.vocab_domain, 'npz')
        chunks = self.get_chunks(filepath)
        if self.prefetch > 0 and i % self.n_chunks == 0:
            ni = (i+(self.n_chunks*self.prefetch)) % self.n_items
            nchunkseq = ni // self.n_chunks
            filepath = chunk_filename(self.cache_dir, nchunkseq, self.vocab_domain, 'npz')
            download(self.url, self.cache_dir, filepath, sync=False)
        return chunks[i % self.n_chunks]

    def compact(self, start, end):
        return start, end

    def reset(self):
        self.cache = {}
        for chunkpath in self.deque:
            os.remove(f'{self.cache_dir}/{chunkpath}')
        self.deque = deque(maxlen=64)

def build_inputs_for_clm(data, max_length=None):
    if isinstance(max_length, int) and len(data) > max_length:
        data[max_length-1]=data[-1]
        return torch.tensor(data[:max_length].astype(np.int32), dtype=torch.long)
        #return torch.from_numpy(data[:max_length])
    else:
        return torch.tensor(data.astype(np.int32), dtype=torch.long)
        #return torch.from_numpy(data)

class FileDataset(Dataset):
    def __init__(self, tokenizer, filename, max_length=DEFAULT_MAX_LENGTH):
        if isinstance(tokenizer, str):
            tokenizer = load_tokenizer(tokenizer)
        self.chunks = []
        token_counts = []
        start = time.time()
        with zopen(filename) as f:
            line = f.readline()
            while line:
                line = line.strip()
                if line.startswith('{'):
                    d = json.loads(line)
                    if 'out' in d:
                        ids = tokenize_pair(tokenizer, d['in'], d['out'], max_length=max_length)
                    else:
                        ids = tokenize_text(d['text'], max_length=max_length)
                else:
                    ids = tokenize_text(line, max_length=max_length)
                self.chunks.append(np.array(ids, dtype=np.int32))
                token_counts.append(len(ids))
                line = f.readline()
        end = time.time()
        verbose_print(f'Loaded {filename} {(end-start):.3f}s:', stat_tokens(token_counts))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i]

    def compact(self, start, end):
        self.chunks = self.chunks[start:end]
        return 0, end-start

def build_inputs_for_seq2seq(data, max_length=None, target_max_length=None):
    target_max_length = target_max_length or max_length
    eos_id = data[-1]
    indices = np.where(data == SpecialTokenIds.SEP)[0]
    # index = data.tolist().index(SpecialTokenIds.SEP)
    if indices.size > 0:
        index = indices[0]
        inputs = data[:index+1]
        inputs[index] = eos_id
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
        inputs=np.array(inputs, dtype=np.int32)
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


def _parse_split(url, vocab_domain):
    start, end = '', ''
    if '[' in url and url.endswith(']'):
        url, _, split = url.rpartition('[')
        start, end = split[:-1].split(':')
    if '?' in url:
        url, _, vocab_domain = url.rpartition('?')
    return url, vocab_domain, start, end

def _parse_range(start, end, n_items):
    try:
        start = float(start)
        if start < 1:
            start = int(n_items * start)
        else:
            start = min(int(start), n_items)
        if start < 0:
            start = n_items - start
    except ValueError:
        start = 0
    try:
        end = float(end)
        if end < 1:
            end = int(n_items*end)
        else:
            end = min(int(end), n_items)
        if end < 0:
            end = n_items - end
    except ValueError:
        end = n_items
    if end - start > 0:
        return start, end
    return end, start

class Indexer(Dataset):
    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        self.count = 0
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.count
        self.count = (self.count + 1) % self.length
        return self.dataset[self.offset+idx]

def _make_mixer(datasets: Indexer):
    lens = [len(ds) for ds in datasets]
    total = sum(lens)
    mixer_base = (total // min(lens))+1
    lens = [int((dlen * mixer_base) / total) for dlen in lens]
    verbose_print('Mixer:', lens)
    mixer = []
    for dlen, ds in zip(lens, datasets):
        mixer.extend([ds]*dlen)
    random.shuffle(mixer)
    return mixer


class DataComposer(Dataset):
    def __init__(self, urls=[], vocab_domain = 'undefined', cache_dir = '.', tokenizer=None, train=None, max_length=DEFAULT_MAX_LENGTH):
        self.vocab_domain = vocab_domain
        self.cache_dir = f'{cache_dir}/{random_name()}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock_file = f'{self.cache_dir}/lock'
        self.max_length = max_length
        if train == 'seq2seq' or train == 't5':
            self.build_fn = build_inputs_for_seq2seq
        else:
            self.build_fn = build_inputs_for_clm
        self.n_items = 0
        self._load_datasets(urls, tokenizer)

    def _load_datasets(self, urls, tokenizer):
        if isinstance(urls, str):
            urls = urls.split('|')
        self.n_items = 0
        datasets = []
        for i, url in enumerate(urls):
            url, vocab_domain, start, end = _parse_split(url, self.vocab_domain)
            if url.endswith('.jsonl') or url.endswith('.jsonl.gz'):
                if tokenizer is None:
                    raise ValueError('tokenizer must be specified.')
                dataset = FileDataset(tokenizer, url, max_length=self.max_length)
            else:
                dataset = ChunkedDataset(url, vocab_domain=vocab_domain, lock_file=self.lock_file, cache_dir=f'{self.cache_dir}/{i}')
            start, end = _parse_range(start, end, len(dataset))
            self.n_items += (end - start)
            start, end = dataset.compact(start, end)
            datasets.append(Indexer(dataset, start, end-start))
        self.mixer = _make_mixer(datasets)


    def set_build(self, build_fn):
        self.build_fn = build_fn


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.mixer = None
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        print(torch.utils.data.get_worker_info())

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        mix = len(self.mixer)
        item = self.mixer[idx % mix][idx]
        return self.build_fn(item, self.max_length)
