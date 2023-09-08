import os
import time
import random

import json
import shutil
import subprocess
from typing import List

from collections import deque

from filelock import FileLock
import numpy as np
import gzip

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from .papertown_utils import verbose_print
from .papertown_tokenizer import load_tokenizer, get_tokenizer_info

def safe_dir(dir):
    if dir.endswith('/'):
        dir = dir[:-1]
    return dir

_ID = 0
def random_name():
    global _ID
    _ID+= 1
    return f'Cache{_ID-1}'

# 設定ファイル

DEFAULT_MAX_LENGTH = 1024
N_CHUNKS = 4096

DEFAULT_TOKENIZER='kkuramitsu/spm-pt32k'
DEFAULT_VOCAB_DOMAIN='kogi'
DEFAULT_CACHE_DIR = safe_dir(os.environ.get('PT_CACHE_DIR', '.'))

def zopen(filepath):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    else:
        return open(filepath, 'r')

def _read_text_from_line(line, key, sep=None):
    if key is None and sep is None:
        return line.rstrip()
    d = json.loads(line)
    if sep is not None and 'out' in d:
        # 後方互換性のための
        return f"{d['in']}{sep}{d['out']}"
    return d[key]

def _remove_heading_nL(s):
    while s.startswith('<nL>'):
        s = s[4:]
    return s


def chunk_filename(dir:str, chunkseq:int, prefix:str, file_ext:str, mkdir=True):
    dir = f"{dir}/{(chunkseq//100):04d}"
    if mkdir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return f"{dir}/{prefix}{(chunkseq%100):02d}.{file_ext}"


def save_chunk(dir, chunkseq, prefix, file_ext, chunks):
    filename = chunk_filename(dir, chunkseq, prefix, file_ext)
    if filename.endswith('.npz'):
        np.savez_compressed(filename, *chunks)
    # elif filename.endswith('.jsonl.gz'):
    #     with gzip.open(filename, 'wt') as w:
    #         for chunk in chunks:
    #             print(json.dumps(chunk, ensure_ascii=False), file=w)

def load_chunk_npz(filename):
    npz = np.load(filename)
    return [npz[n] for n in npz.files]

def load_chunk(dir: str, chunkseq: int, prefix: str, file_ext: str):
    filename = chunk_filename(dir, chunkseq, prefix, file_ext, mkdir=False)
    if filename.endswith(".npz"):
        chunks = load_chunk_npz(filename)
    # elif filename.endswith("jsonl") or filename.endswith('jsonl.gz'):
    #     chunks=[]
    #     with zopen(filename) as f:
    #         for line in f.readlines():
    #             d = json.loads(line)
    #             chunks.append(d)
    # else:
    #     chunks=[]
    #     with zopen(filename) as f:
    #         for line in f.readlines():
    #             chunks.append(line.strip().replace('<nl>', '<nL>'))
    return chunks

# def map_chunk(dir, chunkseq, prefix, map_fn):
#     chunks = load_chunk(dir, chunkseq, '', 'jsonl.gz')
#     chunks = [map_fn(chunk) for chunk in chunks]
#     save_chunk(dir, chunkseq, prefix, prefix, 'npz', chunks)
#     return [len(chunk) for chunk in chunks]

def _tokenize_block_simply(blocks: List[List[int]], tokens: List[int], fill=None, block_size=256, overlap=False):
    # とりあえず、シンプルにブロックを分割する
    for i in range(0, len(tokens) - block_size + 1, block_size):  
        segmented = tokens[i : i + block_size]
        blocks.append(segmented)
    remaining = len(tokens) % block_size
    # 最後の分割が揃っていればおしまい
    if remaining == 0:
        return fill
    # オーバーラップが有効ならオーバラップを検討する
    if overlap and len(tokens) > block_size:
        blocks.append(tokens[-block_size:])
        return fill
    if fill is None:
        return tokens[-remaining:]
    fill = tokens[-remaining:] + fill
    if len(fill) >= block_size:
        segmented = fill[:block_size]
        blocks.append(segmented)
        fill = fill[block_size:]
    return fill
    
def tokenize_block_by_line(tokenizer, blocks, text, fill=None, block_size=256, nL_id = None):
    if '<nL><nL>' in text:
        lines = text.split('<nL><nL>')
        NL = [nL_id, nL_id]
    else:
        lines = text.split('<nL>')
        NL = [nL_id]
    buffer = []
    for line in lines:
        overlap = len(buffer) > (block_size // 2)
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))+NL
        buffer.extend(tokens)
        if len(buffer) >= block_size:
            _, remain = _tokenize_block_simply(blocks, buffer, fill=None, block_size=block_size)
            buffer = [] if remain is None else remain
            if overlap and len(tokens) < block_size:
                buffer = tokens
    tokens = tokenizer.build_inputs_with_special_tokens(buffer)
    return _tokenize_block_simply(blocks, tokens, fill, block_size, overlap=True)

def tokenize_block(tokenizer, text, fill=None, block_size=256, by_line=False, overlap=False):
    blocks = []
    text = text.replace('\r\n', '<nL>').replace('\n', '<nL>')
    if by_line:
        nL_id=tokenizer.newline_token_id
        if nL_id is not None:
            fill = tokenize_block_by_line(tokenizer, blocks, text, fill, block_size, nL_id=nL_id)
            return blocks, fill
    tokens = tokenizer.encode(text)
    fill = _tokenize_block_simply(blocks, tokens, fill, block_size, overlap=overlap)
    return blocks, fill

# def tokenize_text(tokenizer, text, max_length=DEFAULT_MAX_LENGTH):
#     return tokenizer.encode(text, truncation=True, max_length=max_length)

# def tokenize_pair(tokenizer, text, text2, max_length=DEFAULT_MAX_LENGTH, target_max_length=None):
#     assert tokenizer.sep_token_id is not None
#     target_max_length = target_max_length or max_length
#     text = tokenizer.encode(text, truncation=True, max_length=max_length)
#     text[-1] = tokenizer.sep_token_id
#     text2 = tokenizer.encode(text2, truncation=True, max_length=target_max_length)
#     return text+text2

def stat_tokens(counts):
    if len(counts) == 0:
        return {'total': 0}
    data = np.array(counts)
    return {
        'total': int(np.sum(data)),
        'mean': float(np.mean(data)),
        'std': float(np.var(data)) ** 0.5,
        'max': int(np.max(data)),
        '75%': int(np.percentile(data, 75)),
        'median': int(np.median(data)),
        '25%': int(np.percentile(data, 25)),
        'min': int(np.min(data)),
    }

class DatasetStore(object):
    def __init__(self, dir, vocab_domain=DEFAULT_VOCAB_DOMAIN, **kwargs):
        self.dir = safe_dir(dir)
        self.config = dict(kwargs)
        self.tokenizer = self.config.get("tokenizer", None)
        self.vocab_domain = vocab_domain
        if self.tokenizer is None:
            verbose_print(f'tokenizer is unset and using default {DEFAULT_TOKENIZER}')
            self.tokenizer = load_tokenizer(DEFAULT_TOKENIZER)
            self.vocab_domain = 'kogi'
        self.config['tokenizer'] = get_tokenizer_info(self.tokenizer)
        self.block_size = self.config.get("block_size", DEFAULT_MAX_LENGTH)
        self.file_ext = self.config.get("file_ext", "npz")
        self.n_chunks = self.config.get("n_chunks", N_CHUNKS)
        self.chunkseq = 0
        self.bufs = []
        self.n_items = 0
        self.token_counts = []
        self.shuffle = self.config.get("shuffle", True)

    def save_config(self):
        config_file = f"{self.dir}/{self.vocab_domain}config.json"
        with open(config_file, "w") as w:
            json.dump(self.config, w)
    
    def save(self, save_config=True):
        if len(self.bufs) > 0:
            save_chunk(self.dir, self.chunkseq, self.vocab_domain, self.file_ext, self.bufs)
            self.n_items += len(self.bufs)
            if len(self.bufs) == self.n_chunks:
                self.chunkseq += 1
                self.bufs = []
        if save_config:
            self.config.update(dict(
                n_tokens=self.n_items * self.block_size, # 概算
                block_size=self.block_size,
                n_items=self.n_items,
                n_chunks=self.n_chunks,
                chunkseq=self.chunkseq,
                tokens=stat_tokens(self.token_counts)
            ))
            self.n_tokens = self.config['n_tokens'] = self.config['tokens']['total'] # 正確な値
            self.save_config()

    def append(self, block: List[int]):
        self.bufs.append(np.array(block, dtype=np.int32))
        self.token_counts.append(len(block))
        if len(self.bufs) == self.n_chunks:
            self.save(save_config=False)

    def extend(self, blocks: List[List[int]]):
        for block in blocks:
            self.append(block)    

    def upload(self, filename, by_line=False, overlap=True, jsonl_text='text', N=None):
        fill=None
        if not filename.endswith('.jsonl') or not filename.endswith('.jsonl.gz'):
            jsonl_text=None # jsonl でない
        if N:
            from tqdm import tqdm
            pbar = tqdm(total=N, desc=filename)
        with zopen(filename) as f:
            line = f.readline()
            c=1
            while line:
                line = _read_text_from_line(line, jsonl_text)
                line = _remove_heading_nL(line)
                blocks, fill = tokenize_block(self.tokenizer, line, fill, self.block_size, by_line=by_line, overlap=overlap)
                self.extend(blocks)
                line = f.readline()
                c+=1
                if N: 
                    pbar.update()
                    if c > N: break
        if N:
            pbar.close()
        self.save()
        verbose_print(f'Tokens: {self.n_tokens:,} Items: {self.n_items:,}')

    # def append_block(self, text, fill=None, block_size=256, by_line=False, shuffle=True):
    #     blocks, fill = tokenize_block(self.tokenizer, text, fill, block_size, by_line=by_line, shuffle=shuffle)
    #     for b in blocks:
    #         self.append(b)
    #     return fill

    # def append_dict(self, d: List[int]):
    #     self.bufs.append(d)
    #     if len(self.bufs) == self.n_chunks:
    #         self.save(save_config=False)

    # def append_text(self, text, block_size=None, max_length=DEFAULT_MAX_LENGTH):
    #     if isinstance(block_size, int):
    #         self.append_block(text, block_size=block_size)
    #     else:
    #         self.append(tokenize_text(self.tokenizer, text, max_length=max_length))

    # def append_pair(self, text, text2, max_length=DEFAULT_MAX_LENGTH, target_max_length=None):
    #     self.append(tokenize_pair(self.tokenizer, text, text2, max_length=max_length))

    # def __len__(self):
    #     # 切り上げ
    #     return (self.n_items + self.n_chunks - 1) // self.n_chunks

    # def __getitem__(self, i):
    #     return chunk_filename(self.cache_dir, i, self.vocab_domain, self.file_ext, mkdir=False)

#    cmd = "aria2c -x5 -o {0} {1}".format(url.split('/')[-1], url)
#    subprocess.call(cmd, shell=True)

def safe_join(dir, file):
    if dir.endswith('/'):
        dir = dir[:-1]
    if file.startswith('/'):
        file = file[1:]
    return f'{dir}/{file}'

def download(url, dir, local_file, sync=True):
    file=local_file[len(dir)+1:]
    remote_file = safe_join(url, file)
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

# ChunkedDataset

class _DummyFileLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

def _FileLock(lockfile: str):
    # lockがNoneなら何もしない
    return _DummyFileLock() if lockfile is None else FileLock(lockfile)


class ChunkedDataset(Dataset):
    def __init__(self, url, vocab_domain, lock_file=None, cache_dir=".", prefetch=3, **kwargs):
        self.url = safe_dir(url)
        self.vocab_domain = vocab_domain
        self.cache_dir = f'{safe_dir(cache_dir)}/{random_name()}'
        self.lock_file = lock_file
        self.config = self.load_config(kwargs)
        self.file_ext = self.config.get("file_ext", "npz")
        self.n_tokens = self.config.get('n_tokens', 0)
        self.n_items = self.config.get("n_items", 0)
        self.n_chunks = self.config.get("n_chunks", N_CHUNKS)
        self.prefetch=prefetch
        self.queue = deque(maxlen=64)
        self.cache = {}

    def load_config(self, kwargs: dict):
        with _FileLock(self.lock_file):
            os.makedirs(self.cache_dir, exist_ok=True)
            config_file = f"{self.cache_dir}/{self.vocab_domain}config.json"
            if self.url and not os.path.exists(config_file):
                download(self.url, self.cache_dir, config_file)
        try:
            with open(config_file) as f:
                config = json.load(f)
        except Exception as e:
            verbose_print(f'Error!!: unable to read url={self.url} or vocab_domain={self.vocab_domain}, because of {e}')
            config = dict(n_items=0, n_tokens=0)
        config.update(kwargs)
        return config
    
    def __len__(self):
        return self.n_items

    def get_chunks(self, filepath):
        if filepath in self.cache:
            return self.cache[filepath]
        with _FileLock(self.lock_file):
            if not os.path.exists(filepath):
                download(self.url, self.cache_dir, filepath)
        with _FileLock(self.lock_file):
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

class FileDataset(Dataset):
    def __init__(self, tokenizer, filename, max_length=DEFAULT_MAX_LENGTH):
        if isinstance(tokenizer, str):
            tokenizer = load_tokenizer(tokenizer)
        self.chunks = []
        token_counts = []
        jsonl_text='text'
        if not filename.endswith('.jsonl') or not filename.endswith('.jsonl.gz'):
            jsonl_text=None # jsonl でない
        start = time.time()
        with zopen(filename) as f:
            line = f.readline()
            while line:
                line = _read_text_from_line(line, key=jsonl_text, sep='<sep>')
                ids = tokenizer.encode(line, truncation=True, max_length=max_length)
                self.chunks.append(np.array(ids, dtype=np.int32))
                token_counts.append(len(ids))
                line = f.readline()
        end = time.time()
        stat = stat_tokens(token_counts)
        verbose_print(f'Loaded {filename} {(end-start):.3f}s:', stat)
        self.n_tokens = stat['total']

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i]

    def compact(self, start, end):
        length = end - start
        if length == len(self.chunks):
            return 0, length
        self.n_tokens  = self.n_tokens * length // len(self.chunks)
        self.chunks = self.chunks[start:end]
        return 0, length

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
    except ValueError as e:
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
        self.n_tokens = dataset.n_tokens

    def get_num_of_tokens(self):
        if len(self.dataset) == self.length:
            return self.n_tokens
        return self.n_tokens * self.length // len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.count
        self.count = (self.count + 1) % self.length
        return self.dataset[self.offset+idx]
    

def load_dataset(url, vocab_domain=DEFAULT_VOCAB_DOMAIN, tokenizer=None, lock_file=None, cache_dir='.'):
    url, vocab_domain, start, end = _parse_split(url, vocab_domain)
    if url.endswith('.gz') or url.endswith('.jsonl') or url.endswith('.txt'):
        if tokenizer is None:
            raise ValueError('tokenizer must be specified.')
        dataset = FileDataset(tokenizer, url)
    else:
        dataset = ChunkedDataset(url, vocab_domain=vocab_domain, lock_file=lock_file, cache_dir=cache_dir)
    start, end = _parse_range(start, end, len(dataset))
    start, end = dataset.compact(start, end)
    return Indexer(dataset, start, end-start)


def _make_mixer(datasets: List[Indexer]):
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


def build_inputs_for_clm(data, max_length):
    # if isinstance(max_length, int) and len(data) > max_length:
    #     return torch.tensor(data[:max_length].astype(np.int32), dtype=torch.long)
    # else:
    #     return torch.tensor(data.astype(np.int32), dtype=torch.long)
    return torch.tensor(data[:max_length].astype(np.int32), dtype=torch.long)

class DataComposer(Dataset):
    def __init__(self, urls=[], 
                 max_length=DEFAULT_MAX_LENGTH, build_fn=build_inputs_for_clm,
                 vocab_domain = DEFAULT_VOCAB_DOMAIN, tokenizer=None, 
                 cache_dir = DEFAULT_CACHE_DIR, use_filelock=True):
        self.max_length = max_length
        self.vocab_domain = vocab_domain
        self.cache_dir = f'{safe_dir(cache_dir)}/{random_name()}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock_file = f'{self.cache_dir}/lock' if use_filelock else None
        self.build_fn = build_fn
        self._load_datasets(urls, tokenizer)

    def _load_datasets(self, urls, tokenizer):
        if isinstance(urls, str):
            urls = urls.split('|')
        self.n_items = 0
        self.n_tokens = 0
        datasets = []
        for i, url in enumerate(urls):
            ds = load_dataset(url, vocab_domain=self.vocab_domain, tokenizer=tokenizer,
                              lock_file=self.lock_file, cache_dir=f'{self.cache_dir}/{i}')
            self.n_items += len(ds)
            self.n_tokens += ds.get_num_of_tokens()
            datasets.append(ds)
        self.mixer = _make_mixer(datasets)
        verbose_print(f'Total tokens: {self.n_tokens:,}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.mixer = None
        if os.path.isdir(self.cache_dir):
            verbose_print(f'Cleaning up {self.cache_dir} ...')
            shutil.rmtree(self.cache_dir)

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        mix = len(self.mixer)
        item = self.mixer[idx % mix][idx]
        return self.build_fn(item, self.max_length)


## Seq2Seq

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
        "input_ids": torch.tensor(inputs.astype(np.int64), dtype=torch.long),
        # "attention_mask": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels.astype(np.int64), dtype=torch.long),
    }
