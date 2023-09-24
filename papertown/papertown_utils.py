import os

def safe_dir(dir):
    if dir.endswith('/'):
        dir = dir[:-1]
    return dir

def safe_join(dir, file):
    if dir.endswith('/'):
        dir = dir[:-1]
    if file.startswith('/'):
        file = file[1:]
    return f'{dir}/{file}'

DEFAULT_TOKENIZER = os.environ.get('PT_TOKENIZER', 'kkuramitsu/spm-pt32k')
DEFAULT_VERSION='v1_'
DEFAULT_CACHE_DIR = safe_dir(os.environ.get('PT_CACHE_DIR', '.'))



def verbose_print(*args, **kwargs):
    """
    PaperTown 用のデバッグプリント
    """
    print('📃', *args, **kwargs)

