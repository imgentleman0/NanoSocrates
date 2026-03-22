from collections import Counter
from contextlib import nullcontext
from typing import Dict, List
import torch
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import sacrebleu
from tqdm import trange


def strip_bos_eos_pad(ids, tokenizer):  # Used to strip the special tokens from autoregressive output
    bos_id = tokenizer.convert_tokens_to_ids('[BOS]')
    eos_id = tokenizer.convert_tokens_to_ids('[EOS]')
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    to_drop = {x for x in (bos_id, eos_id, pad_id) if x is not None}

    return [t for t in ids if t not in to_drop]


# Fuction called in meteor_full to ensure that NTLK data packets (wordnet and omw-1.4) are present.
def _ensure_nltk_data():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        try:
            nltk.download('omw-1.4', quiet=True)
        except Exception:
            pass


# We define this function to calculate easily RougeL as metric for RDF2Text task
def rouge_l(candidate_text: str, reference_text: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, candidate_text)
    return scores['rougeL'].fmeasure


# We define this function to calculate easily METEOR as a metric for RDF2Text task
def meteor_full(candidate_text: str, reference_text: str) -> float:
    _ensure_nltk_data()
    hyp_tok = candidate_text.strip().split()
    ref_tok = reference_text.strip().split()
    return meteor_score([ref_tok], hyp_tok)


# We use this functions to print the validation results as a tab
def print_val(results: Dict[str, Dict[str, float]], epoch: int) -> None:
    def f(x):
        return f"{x:.4f}"

    print("\n" + "=" * 72)
    print(f"[Validation] Epoch {epoch}")
    print("-" * 72)
    if "rdf2text" in results:
        r = results["rdf2text"]
        print(
            f"RDF2Text   | BLEU4: {f(r.get('BLEU4', 0))} | ROUGE-L: {f(r.get('ROUGE-L', 0))} | METEOR: {f(r.get('METEOR', 0))}")
    if "text2rdf" in results:
        r = results["text2rdf"]
        print(f"Text2RDF   | P: {f(r.get('precision', 0))} | R: {f(r.get('recall', 0))} | F1: {f(r.get('f1', 0))}")
    if "rdf_completion_1" in results:
        r = results["rdf_completion_1"]
        print(f"Compl-1    | Sample_acc: {f(r.get('sample_accuracy', 0))} | Token_acc: {f(r.get('token_accuracy', 0))}")
    if "rdf_completion_2" in results:
        r = results["rdf_completion_2"]
        print(f"Compl-2    | P: {f(r.get('precision', 0))} | R: {f(r.get('recall', 0))} | F1: {f(r.get('f1', 0))}")
    print("=" * 72 + "\n")


# We use this function to aggregate the metrics for the RDF2Text metrics.
def aggregate_text_metrics(
        ref_texts: List[str],
        hyp_texts: List[str],
        skip_meteor: bool = False
        # Flag used to skip METEOR to save time in evaluation, as is the heaviest metric to compute. If skipped, it is
        # automatically set to 0. With the current setting, it is calculated every 20 epochs.
) -> Dict[str, float]:
    # It gets the minimum length between ref_texts and hyp_texts to allineate them
    n = min(len(ref_texts), len(hyp_texts))
    if n == 0:  # It means that one between ref_texts and hyp_texts is empty
        return {"BLEU4": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0}

    refs = ref_texts[:n]
    hyps = hyp_texts[:n]

    # BLEU (sacrebleu)
    try:
        b = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a", lowercase=False, use_effective_order=True)
    except TypeError:
        b = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a", lowercase=False)
    bleu = b.score / 100.0

    # ROUGE-L
    rouge_vals = [rouge_l(h, r) for r, h in zip(refs, hyps)]
    rouge = sum(rouge_vals) / len(rouge_vals) if rouge_vals else 0.0

    # METEOR
    if skip_meteor:  # Flag described before
        meteor = 0.0
    else:
        try:
            meteor_vals = [meteor_full(h, r) for r, h in zip(refs, hyps)]
            meteor = sum(meteor_vals) / len(meteor_vals) if meteor_vals else 0.0
        except Exception:
            # In case of error, don't stop the evaluation
            meteor = 0.0

    return {"BLEU4": bleu, "ROUGE-L": rouge, "METEOR": meteor}


# We define this function to calculate precision, recall and F1-score as metrics in the Text2RDF and RDF Completion 2 tasks
def prf(pred_ids, ref_ids):
    cp = Counter(pred_ids)
    cr = Counter(ref_ids)
    tp = sum((
                     cp & cr).values())  # True positives: sum of common occurrences (element-wise minimum) between predicted and reference Counters
    p = (tp / sum(cp.values())) if cp else 0.0
    r = (tp / sum(cr.values())) if cr else 0.0
    f1 = (2 * p * r / (p + r)) if (p > 0 and r > 0) else 0.0
    return tp, sum(cp.values()), sum(cr.values()), p, r, f1


@torch.no_grad()
# With this function, we compute the validation loss, saved in learning_curves.csv with the train loss and the metrics for
# the tasks
def compute_val_loss(model, val_loader, tokenizer, device, vocab_size, max_batches=100):
    model.eval()
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.0).to(device)

    total, count = 0.0, 0
    for b, batch in enumerate(val_loader, 1):
        if max_batches and b > max_batches:
            break
        enc_in = batch['encoder_input'].to(device, non_blocking=True)
        dec_in = batch['decoder_input'].to(device, non_blocking=True)
        enc_m = batch['encoder_mask'].to(device, non_blocking=True).bool()
        dec_m = batch['decoder_mask'].to(device, non_blocking=True).bool()
        labels = batch['label'].to(device, non_blocking=True)

        logits = model.project(model.decode(model.encode(enc_in, enc_m), enc_m, dec_in, dec_m))
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        total += loss.item()
        count += 1
    return total / max(1, count)

# Utils function used in the sanity check
@torch.no_grad()
def token_accuracy(logits, labels, pad_id):
    # logits: (B, S, V), labels: (B, S)
    pred = logits.argmax(dim=-1)
    mask = (labels != pad_id)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


# Function used to perform sanity check on the model.
def sanity_overfit_one_batch(model, train_dataloader, tokenizer, device, steps=800, lr=1e-3, disable_dropout=True):
    # Helper functions to disable and the restore all the dropouts
    def _disable_all_dropout(m):
        saved = []
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                saved.append((mod, mod.p))
                mod.p = 0.0
        return saved

    def _restore_all_dropout(saved):
        for mod, p in saved:
            mod.p = p

    model.to(device)

    saved_dropouts = _disable_all_dropout(model) if disable_dropout else []

    model.train()


    # We fix only one batch
    batch_fixed = next(iter(train_dataloader))
    batch_fixed = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_fixed.items()}

    # Loss without smoothing
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.0).to(device)

    # Optimizer without weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # AMP only on CUDA
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    amp_ctx = (autocast(dtype=torch.bfloat16) if use_amp else nullcontext())

    try:
        # Training loop always on the same batch
        pbar = trange(steps, desc="Sanity one-batch")
        for step in pbar:
            encoder_input = batch_fixed["encoder_input"]
            decoder_input = batch_fixed["decoder_input"]
            encoder_mask = batch_fixed["encoder_mask"].bool()
            decoder_mask = batch_fixed["decoder_mask"].bool()
            labels = batch_fixed["label"]

            with amp_ctx:
                enc_out = model.encode(encoder_input, encoder_mask)
                dec_out = model.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
                logits = model.project(dec_out)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Token accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                valid = (labels != pad_id)
                acc = ((pred == labels) & valid).sum().float() / valid.sum().float()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.1f}%")

        # We take an example to see the predicition
        with torch.no_grad():
            enc_in = batch_fixed["encoder_input"]
            dec_in = batch_fixed["decoder_input"]
            enc_m = batch_fixed["encoder_mask"].bool()
            dec_m = batch_fixed["decoder_mask"].bool()
            with amp_ctx:
                logits = model.project(model.decode(model.encode(enc_in, enc_m), enc_m, dec_in, dec_m))
            pred_ids = logits.argmax(dim=-1)  # (B, S)

        def strip_after(seq, stop_ids, pad_id):
            out = []
            for tok in seq:
                if tok in stop_ids:
                    break
                if tok != pad_id:
                    out.append(tok)
            return out

        eos_id = tokenizer.convert_tokens_to_ids('[EOS]')
        stop_ids = {x for x in [pad_id, eos_id] if x is not None}

        i = 0
        src_seq = batch_fixed["encoder_input"][i].detach().cpu().tolist()
        tgt_seq = batch_fixed["label"][i].detach().cpu().tolist()
        pred_seq = pred_ids[i].detach().cpu().tolist()

        src_seq_clean = strip_after(src_seq, stop_ids, pad_id)
        tgt_seq_clean = strip_after(tgt_seq, stop_ids, pad_id)
        pred_seq_clean = strip_after(pred_seq, stop_ids, pad_id)

        try:
            src_text = tokenizer.decode(src_seq_clean, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            src_text = str(src_seq_clean)
        try:
            tgt_text = tokenizer.decode(tgt_seq_clean, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            tgt_text = str(tgt_seq_clean)
        try:
            pred_text = tokenizer.decode(pred_seq_clean, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            pred_text = str(pred_seq_clean)

        print("\n" + "-" * 80)
        print("Sanity check on 1 batch — example 0")
        print(f"SOURCE:    {src_text}")
        print(f"TARGET:    {tgt_text}")
        print(f"PREDICTED: {pred_text}")
        print("-" * 80)

        print(model)

    finally:
        # Restores the dropouts as they were
        if disable_dropout:
            _restore_all_dropout(saved_dropouts)