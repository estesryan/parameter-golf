import torch
import torch.nn.functional as F
from collections import defaultdict
import math
import importlib.util
import argparse

def load_model(script_path, checkpoint_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.GPT(mod.Hyperparameters())
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, mod

def build_ngram_counts(data, n=2):
    counts = defaultdict(lambda: defaultdict(int))
    context_counts = defaultdict(int)

    for seq in data:
        for i in range(n - 1, len(seq)):
            ctx = tuple(seq[i - (n - 1):i])
            tok = seq[i]
            counts[ctx][tok] += 1
            context_counts[ctx] += 1

    return counts, context_counts

def ngram_logprob(counts, context_counts, ctx, tok, vocab_size=1024, alpha=1.0):
    c = counts.get(ctx, {})
    total = context_counts.get(ctx, 0)
    return math.log((c.get(tok, 0) + alpha) / (total + alpha * vocab_size))

def evaluate(model, data, bigram_counts, bigram_ctx_counts,
             trigram_counts, trigram_ctx_counts):

    total = 0
    loss_model = 0
    loss_bigram = 0
    loss_trigram = 0

    device = next(model.parameters()).device

    for seq in data:
        x = torch.tensor(seq[:-1], dtype=torch.long)[None, :].to(device)
        y = torch.tensor(seq[1:], dtype=torch.long)[None, :].to(device)

        with torch.no_grad():
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=-1)

        for t in range(x.shape[1]):
            target = y[0, t].item()

            # model
            loss_model -= log_probs[0, t, target].item()

            # bigram
            ctx2 = (x[0, t].item(),)
            loss_bigram -= ngram_logprob(bigram_counts, bigram_ctx_counts, ctx2, target)

            # trigram
            if t >= 1:
                ctx3 = (x[0, t-1].item(), x[0, t].item())
                loss_trigram -= ngram_logprob(trigram_counts, trigram_ctx_counts, ctx3, target)
            else:
                loss_trigram -= math.log(1.0 / 1024)

            total += 1

    return {
        "model_bpb": loss_model / total / math.log(2),
        "bigram_bpb": loss_bigram / total / math.log(2),
        "trigram_bpb": loss_trigram / total / math.log(2),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-script", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-tokens", type=int, default=200000)
    args = parser.parse_args()

    print("Loading model...")
    model, mod = load_model(args.model_script, args.checkpoint)

    print("Loading validation data...")
    val_loader = mod.get_val_loader()

    data = []
    total = 0

    for batch in val_loader:
        tokens = batch.tolist()
        for seq in tokens:
            data.append(seq)
            total += len(seq)
            if total > args.val_tokens:
                break
        if total > args.val_tokens:
            break

    print("Building bigram...")
    bigram_counts, bigram_ctx_counts = build_ngram_counts(data, n=2)

    print("Building trigram...")
    trigram_counts, trigram_ctx_counts = build_ngram_counts(data, n=3)

    print("Evaluating...")
    results = evaluate(
        model,
        data,
        bigram_counts,
        bigram_ctx_counts,
        trigram_counts,
        trigram_ctx_counts,
    )

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\n=== GAPS ===")
    print(f"model - bigram:  {results['model_bpb'] - results['bigram_bpb']:.4f}")
    print(f"model - trigram: {results['model_bpb'] - results['trigram_bpb']:.4f}")

if __name__ == "__main__":
    main()