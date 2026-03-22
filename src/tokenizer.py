#!/usr/bin/env python3
# train_tokenizer_from_all.py
import json, re
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


# We initalize the parameters for the trainer
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"  # Path containing the dataset
ALL_PATH = DATA_DIR / "dataset.jsonl"
ST_PATH  = DATA_DIR / "special_tokens.txt"
OUT_DIR  = ROOT / "tokenizer"
VOCAB_SIZE = 16000
MIN_FREQ = 2

# Function used to load special tokens from the text file created by database_creator.py
def load_special_tokens(p: Path):
    toks = [t.strip() for t in p.read_text(encoding="utf-8").splitlines() if t.strip()]
    print(f"Special tokens: {len(toks)}")
    return toks

def normalize_line(s: str) -> str:
    s = s.strip() # Removes spaces at the start and at the end of the string
    s = re.sub(r"\s+", " ", s) # Replaces anu sequence of spaces with a single space
    return s

def stream_all_fields(all_path: Path):
    with all_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue # Skips empty lines
            obj = json.loads(line)
            # dataset.jsonl has always only "input" e "target"
            inp = obj.get("input")
            tgt = obj.get("target")
            if isinstance(inp, str) and inp:
                yield inp # Returns one-per-time to the calling stream
            if isinstance(tgt, str) and tgt:
                yield tgt


def build_corpus(all_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.txt"
    seen = set()
    n = 0
    with corpus_path.open("w", encoding="utf-8") as out:
        for s in stream_all_fields(all_path):
            s = normalize_line(s)
            if s in seen:
                continue  # To avoid duplicates
            seen.add(s)
            out.write(s + "\n")
            n += 1
    print(f"Corpus lines: {n}, saved to {corpus_path}")
    return corpus_path

# This function trains byte-level BPE tokenizer, as asked in the exam's track
def train_tokenizer(corpus_path: Path, special_tokens, out_dir: Path):
    tok = Tokenizer(BPE(unk_token="[UNK]")) # We initialize a byte-level BPE tokenizer with [UNK]
    # as unkwown token
    tok.pre_tokenizer = ByteLevel(use_regex=True) # Operates at byte granularity
    tok.decoder = ByteLevelDecoder()

    # We configure the BpeTrainer
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"] + special_tokens,
        show_progress=True,
    )
    tok.train([str(corpus_path)], trainer) # We train the model from the corpus we built before
    tok.post_processor = TemplateProcessing(single="$A", pair="$A $B", special_tokens=[]) # We add manually the special
    # tokens for start and end of sequence in the model script
    tok_path = out_dir / "tokenizer.json"
    tok.save(str(tok_path)) # We save the tokenizers json artifact
    print(f"Saved tokenizer.json to {tok_path}")

    # We wrap as a Hugging Face PreTrainedTokenizerFast, loading from the tokenizer.json
    # and declaring the special tokens for the Transformer
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_path),
        bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]",
        additional_special_tokens=special_tokens, # Redundant if the special tokens are already set, but it is safe,
        # as it ensures they end up in the vocab.
    )
    fast.add_special_tokens({"additional_special_tokens": special_tokens})
    fast.save_pretrained(str(out_dir)) # We save the pretrained tokenizer in the out_dir
    print(f"Saved HF tokenizer to {out_dir}")

if __name__ == "__main__":
    assert ALL_PATH.exists(), f"I don't find {ALL_PATH}"
    assert ST_PATH.exists(),  f"I don't find {ST_PATH}"
    special = load_special_tokens(ST_PATH)
    corpus = build_corpus(ALL_PATH, OUT_DIR)
    train_tokenizer(corpus, special, OUT_DIR)
