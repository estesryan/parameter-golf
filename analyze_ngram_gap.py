import argparse
import importlib.util
import math
from collections import defaultdict

import torch
import torch.nn.functional as F


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(mod):
    args = mod.Hyperparameters()
    model = mod.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        logit_sharpen=args.logit_sharpen,
        rope_partial_dims=args.rope_partial_dims,
        encoder_layer_frac=args.encoder_layer_frac,
        bigram_rank=args.bigram_rank,
    )
    return model, args


def load_model(script_path, checkpoint_path):
    mod = load_module(script_path)
    model, args = build_model(mod)
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, mod, args


def build_ngram_counts(token_ids, n=2):
    counts = defaultdict(lambda: defaultdict(int))
    ctx_counts = defaultdict(int)
    for i in range(n - 1, len(token_ids)):
        ctx = tuple(token_ids[i - (n - 1):i])
        tok = token_ids[i]
        counts[ctx][tok] += 1
        ctx_counts[ctx] += 1
    return counts, ctx_counts


def ngram_logprob(counts, ctx_counts, ctx, tok, vocab_size=1024, alpha=1.0):
    row = counts.get(ctx, {})
    total = ctx_counts.get(ctx, 0)
    return math.log((row.get(tok, 0) + alpha) / (total + alpha * vocab_size))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-script", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokens", type=int, default=2_000_000)
    p.add_argument("--alpha", type=float, default=1.0)
    args_cli = p.parse_args()

    print("Loading model...")
    model, mod, hp = load_model(args_cli.model_script, args_cli.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("Loading validation tokens...")
    val_tokens = mod.load_validation_tokens(hp.val_files, hp.train_seq_len)
    token_ids = val_tokens[: args_cli.tokens].tolist()

    print("Building bigram counts...")
    bigram_counts, bigram_ctx_counts = build_ngram_counts(token_ids, n=2)

    print("Building trigram counts...")
    trigram_counts, trigram_ctx_counts = build_ngram_counts(token_ids, n=3)

    print("Evaluating model and n-gram baselines...")
    total = 0
    loss_model = 0.0
    loss_bigram = 0.0
    loss_trigram = 0.0

    seq_len = hp.train_seq_len
    usable = ((len(token_ids) - 1) // seq_len) * seq_len
    token_ids = token_ids[: usable + 1]

    with torch.no_grad():
        for start in range(0, usable, seq_len):
            chunk = token_ids[start : start + seq_len + 1]
            x_list = chunk[:-1]
            y_list = chunk[1:]

            x = torch.tensor(x_list, dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(y_list, dtype=torch.long, device=device).unsqueeze(0)

            full_logits, _bigram = model._compute_logits(x)
            log_probs = F.log_softmax(full_logits.float(), dim=-1)

            for t in range(seq_len):
                target = y_list[t]
                loss_model -= log_probs[t, target].item()

                ctx2 = (x_list[t],)
                loss_bigram -= ngram_logprob(
                    bigram_counts, bigram_ctx_counts, ctx2, target,
                    vocab_size=hp.vocab_size, alpha=args_cli.alpha
                )

                if t >= 1:
                    ctx3 = (x_list[t - 1], x_list[t])
                    loss_trigram -= ngram_logprob(
                        trigram_counts, trigram_ctx_counts, ctx3, target,
                        vocab_size=hp.vocab_size, alpha=args_cli.alpha
                    )
                else:
                    loss_trigram -= math.log(1.0 / hp.vocab_size)

                total += 1

    model_bpb = loss_model / total / math.log(2.0)
    bigram_bpb = loss_bigram / total / math.log(2.0)
    trigram_bpb = loss_trigram / total / math.log(2.0)

    print("\n=== RESULTS ===")
    print(f"model_bpb:   {model_bpb:.4f}")
    print(f"bigram_bpb:  {bigram_bpb:.4f}")
    print(f"trigram_bpb: {trigram_bpb:.4f}")

    print("\n=== GAPS ===")
    print(f"model - bigram:  {model_bpb - bigram_bpb:+.4f}")
    print(f"model - trigram: {model_bpb - trigram_bpb:+.4f}")


if __name__ == "__main__":
    main()