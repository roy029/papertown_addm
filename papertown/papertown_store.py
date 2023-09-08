import os
import argparse
from papertown import DatasetStore, load_tokenizer

def setup_store():
    parser = argparse.ArgumentParser(description="pt_build_store - duplicate model")
    parser.add_argument("files", type=str, nargs="+", help="files")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--tokenizer_path", default='kkuramitsu/spm-pt32k')
    parser.add_argument("--vocab_domain", default='kogi')
    parser.add_argument("--store_path", default="store")
    parser.add_argument("--num_works", default=0)
    parser.add_argument("-N", default=None)
    hparams = parser.parse_args()  # hparams になる
    return hparams

def main_store():
    hparams = setup_store()
    tokenizer = load_tokenizer(hparams.tokenizer_path)
    store = DatasetStore(tokenizer=tokenizer, 
                         vocab_domain=hparams.vocab_domain, 
                         block_size=hparams.block_size, 
                         dir=hparams.store_path)
    store.upload(filename=hparams.files[0], N=hparams.N)

if __name__ == "__main__":  # わかります
    main_store()

