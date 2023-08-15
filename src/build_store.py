import os
import argparse
import json, gzip, shutil
from papertown import StoreDataset, load_tokenizer, get_tokenizer_info

def setup():
    parser = argparse.ArgumentParser(description="pt_build_store - duplicate model")
    parser.add_argument("files", type=str, nargs="+", help="jsonl files")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tokenizer_path", default='kkuramitsu/spm-pt32k')
    parser.add_argument("--output_path", default="ptstore")
    hparams = parser.parse_args()  # hparams になる
    return hparams


def zopen(filepath):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    else:
        return open(filepath, 'r')

def build_store_from_jsonl(*files, 
                           base='ptstore', 
                           tokenizer_path='kkuramitsu/spm-pt16k', 
                           max_length=None):
    tokenizer = load_tokenizer(tokenizer_path)
    if os.path.exists(base):
        shutil.rmtree(base)
    store = StoreDataset(tokenizer, cache_dir=base)
    for file in files:
        with zopen(file) as f:
            for line in f.readlines():
                d = json.loads(line)
                if 'out' in d:
                    store.append_pair(d['in'], d['out'], max_length=max_length)
                elif 'in' in d:
                    store.append_text(d['in'], max_length=max_length)
                else:
                    store.append_text(d['text'], max_length=max_length)
    store.dump()

def main_jsonl():
    hparams = setup()
    build_store_from_jsonl(hparams.files, 
                           base=hparams.out_path,
                           tokenizer_path=hparams.tokenizer_path, 
                            max_length=hparams.max_length)

if __name__ == "__main__":  # わかります
    main_jsonl()
