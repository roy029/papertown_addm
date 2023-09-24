from typing import List

import os
import re
import hashlib

import torch
from transformers import AutoTokenizer
from .papertown_utils import verbose_print

DEFAULT_NL = os.environ.get('PT_NK', '<nL>')
DEFAULT_SEP = os.environ.get('PT_SEP', '<seP>')
DEFAULT_ELLIPSIS = os.environ.get('PT_ELLIPSIS', '<ellipsiS>')

def find_token_id(tokenizer: AutoTokenizer, *token: str)->int:
    ids = tokenizer.convert_tokens_to_ids(token)
    for id in ids:
        if id != tokenizer.unk_token_id:
            return id
    return tokenizer.unk_token_id

def find_newline_token_id(tokenizer: AutoTokenizer):
    return find_token_id(tokenizer, DEFAULT_NL, "<nL>", "<nl>")

def find_sep_token_id(tokenizer: AutoTokenizer):
    return find_token_id(tokenizer, DEFAULT_SEP, "<seP>", "<sep>")

def find_ellipsis_token_id(tokenizer: AutoTokenizer):
    return find_token_id(tokenizer, DEFAULT_ELLIPSIS, "<ellipsis>", "<masK>", "<mask>", "<extra_id_99>")

_EXTRA_IDS = [f'<extra_id_{i}>' for i in range(100)]

def find_extra_ids(tokenizer: AutoTokenizer):
    return tokenizer.convert_tokens_to_ids(_EXTRA_IDS)


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
    return s.replace('\t', '    ').replace('\n', '<nL>')

def post_decode(s):
    return _CapitalizedPattern.sub(_cap_repl,s).replace('<nL>', '\n')


def adapt_tokenizer(tokenizer: AutoTokenizer):
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


def load_tokenizer(tokenizer_path="kkuramitsu/spm-pt32k", adapt=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False, trust_remote_code=True, use_fast=False)
    if adapt:
        adapt_tokenizer(tokenizer)
    return tokenizer
