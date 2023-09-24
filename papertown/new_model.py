import gc
import torch
torch.backends.cuda.matmul.allow_tf32=True
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import numpy as np
from datasets import Dataset

def count_parameters(model)->int:
    """
    モデルのパラメータ数を数える

    model: モデル
    return パラメータ数
    """
    return sum(p.numel() for p in model.parameters())

def format_large_number(num: int)->str:
    """
    大きな数をSI単位系に変換して返す
    """

    if num < 1_000:
        return str(num)
    elif num < 1_000_000:
        return f"{num / 1_000:.1f}K"
    elif num < 1_000_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num < 1_000_000_000_000:
        return f"{num / 1_000_000_000:.1f}G"
    elif num < 1_000_000_000_000_000:
        return f"{num / 1_000_000_000_000:.1f}T"
    elif num < 1_000_000_000_000_000_000:
        return f"{num / 1_000_000_000_000_000:.1f}P"
    else:
        return f"{num / 1_000_000_000_000_000_000:.1f}Exa"

def print_model(model):
    n_parameters=count_parameters(model)
    config = model.config
    print(f'Parameters: {format_large_number(n_parameters)} {n_parameters}', end=' ')
    if hasattr(config, 'max_position_embeddings'):
        print(f"max_length: {config.max_position_embeddings}", end=' ')
        print(f"vocab_size: {config.vocab_size}")
    elif hasattr(config, "n_positions"):
        print(f"max_length: {model.config.n_positions}", end=' ')
        print(f"vocab_size: {config.vocab_size}")

    if hasattr(config, 'hidden_size'):
        print(f"n_dims: {model.config.hidden_size//model.config.num_attention_heads}", end=' ')
        print(f"n_heads: {model.config.num_attention_heads}", end=' ')
        print(f"hidden_size: {model.config.hidden_size}", end=' ')
        print(f"intermediate_size: {model.config.intermediate_size}", end=' ')
        print(f"n_layers: {model.config.num_hidden_layers}")
    elif hasattr(config, 'n_embd'):
        print(f"n_embed: {model.config.n_embd}", end=' ')
        print(f"n_heads: {model.config.n_head}", end=' ')
        print(f"n_layers: {model.config.n_layer}")
    else:
        print(config)

def print_gpu_utilization():
    try:
        from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")
    except:
        pass

def dummy_dataset(max_length, dataset_size=1024):
    seq_len, data_size = max_length

    dummy_data = {
        "input_ids": (np.arange(100, dataset_size*seq_len+100) % 15000).reshape((dataset_size, seq_len))
    }
    ds = Dataset.from_dict(dummy_data)
    ds.set_format("pt")
    return ds

def print_summary(result, use_flash=False):
    m=result.metrics
    print(f"Time: {m['train_runtime']:.2f}", end=' ')
    print(f"Samples/second: {m['train_samples_per_second']:.2f} FlashAttn={use_flash}")
    print(f"Global step: {result.global_step} batch_size: {1024//result.global_step}", end=' ')
    print(f"FLOS: {m['total_flos']} {format_large_number(m['total_flos'])} Loss: {m['train_loss']:.5f}")
    print_gpu_utilization()


def train_model(model, tokenizer, max_length, use_fp16=False, use_flash=False):

    if use_flash:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        auto_find_batch_size=True,  # バッチサイズ自動
        do_eval=False,
        logging_steps=1000,
#        gradient_accumulation_steps=1,
        num_train_epochs=1,
        weight_decay=0.1,
#        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4, #TODO: 論文から探す
#        save_steps=5_000,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dummy_dataset(max_length),
        data_collator=data_collator,
    )
    result = trainer.train()
    print_summary(result, use_flash)
    # モデルを保存 output_path に保存します
    if use_flash:
        model = BetterTransformer.reverse(model)
    output_path='trained'
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_utilization()


# GPT-2

def new_GPT2(max_length=2048, n_dims=512, n_heads=24, n_layers=24, tokenizer='kkuramitsu/spm-pt32k'):
    from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, trust_remote_code=True, use_fast=False)

    config = GPT2Config(
        vocab_size = len(tokenizer),
        bos_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        n_positions=max_length,
        n_ctx=max_length,
        n_embd=n_dims,
        n_head=n_heads,
        n_layer=n_layers,
    )

    model = GPT2LMHeadModel(config)
    print_model(model)
    return model

# GPTNeoX

def new_GPTNeoX(max_length=2048, n_dims=512, n_heads=8, n_layers=24, intermediate_size=1024, tokenizer='kkuramitsu/spm-pt32k'):
    from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, trust_remote_code=True, use_fast=False)

    config = GPTNeoXConfig(
        vocab_size = len(tokenizer),
        pad_token_id = tokenizer.pad_token_id,
        bos_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        max_position_embeddings=max_length, #トークン数
        hidden_size=n_dims * n_heads,
        num_attention_heads = n_heads, #8
        num_hidden_layers = n_layers, #28
        intermediate_size=intermediate_size,
    )

    model = GPTNeoXForCausalLM(config)
    print_model(model)
    return model


## new_Lamma2

def new_Llama2(max_length=2048, n_dims=512, n_heads=8, n_layers=28, intermediate_size=1024, tokenizer='kkuramitsu/spm-pt32k'):
    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, trust_remote_code=True, use_fast=False)

    config = LlamaConfig(
        vocab_size = len(tokenizer),
        pad_token_id = tokenizer.pad_token_id,
        bos_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        max_position_embeddings=max_length, #トークン数
        hidden_size=n_dims * n_heads,
        num_attention_heads = n_heads, #8
        num_hidden_layers = n_layers, #28
        intermediate_size=intermediate_size,
    )

    model = LlamaForCausalLM(config)
    print_model(model)
    return model



