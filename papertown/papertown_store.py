import os
import time
import argparse
from papertown import DatasetStore, DataComposer, load_tokenizer
from tqdm import tqdm

def setup_store():
    parser = argparse.ArgumentParser(description="papertown_store")
    parser.add_argument("files", type=str, nargs="+", help="files")
    parser.add_argument("--tokenizer_path", default='kkuramitsu/spm-pt32k')
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--version", default='v1')
    parser.add_argument("--store_path", default="store")
    parser.add_argument("-N", type=int, default=None)
    parser.add_argument("--num_works", type=int, default=0)
    hparams = parser.parse_args()  # hparams になる
    return hparams

def main_store():
    hparams = setup_store()
    tokenizer = load_tokenizer(hparams.tokenizer_path)
    store = DatasetStore(tokenizer=tokenizer, 
                         version=hparams.version, 
                         block_size=hparams.block_size, 
                         dir=hparams.store_path)
    store.upload(filename=hparams.files[0], N=hparams.N)



def setup_testdata():
    parser = argparse.ArgumentParser(description="papertown_testdata")
    parser.add_argument("--url_list", type=str)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--N", type=int, default=0)
    hparams = parser.parse_args()  # hparams になる
    return hparams

def main_testdata():
    hparams = setup_testdata()
    with DataComposer(url_list=hparams.url_list, block_size=hparams.block_size) as dc:
        print(len(dc))
        if hparams.N == 0:
            return
        if hparams.N == -1:
            hparams.N = len(dc)
        start = time.time()
        for index in tqdm(range(hparams.N), total=hparams.N):
            dc[index]
        end = time.time()
        print(f'Total: {end-start:.1f}s Iterations: {hparams.N:,} {hparams.N/(end-start)}[it/s]')



if __name__ == "__main__":  # わかります
    main_download()

