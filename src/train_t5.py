
from dataclasses import dataclass, field
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

@dataclass
class PapertownArguments():
    tokenizer_path: str = field(
        default='kkuramitsu/spm-pt16k',
        metadata={'help': 'output filename'})

def main():
    parser = HfArgumentParser([Seq2SeqTrainingArguments,PapertownArguments])
    args = parser.parse_args_into_dataclasses()
    print(args[0],args[1])

if __name__ == '__main__':
    main()