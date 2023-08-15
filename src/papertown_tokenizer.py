import re
import hashlib
from transformers import AutoTokenizer #, T5Tokenizer

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