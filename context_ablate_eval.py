#!/usr/bin/env python3
from __future__ import annotations

import io
import math
import os
import sys
import zlib
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

# -----------------------------
# Load user's training module
# -----------------------------
TRAIN_SCRIPT = os.environ.get("TRAIN_SCRIPT", "train_gpt_hailmary_pruned.py")
spec = importlib.util.spec_from_file_location("user_train_mod", TRAIN_SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["user_train_mod"] = mod
spec.loader.exec_module(mod)

Hyperparameters = mod.Hyperparameters
GPT = mod.GPT
CastedLinear = mod.CastedLinear
restore_low_dim_params_to_fp32 = mod.restore_low_dim_params_to_fp32
load_validation_tokens = mod.load_validation_tokens
build_sentencepiece_luts = mod.build_sentencepiece_luts
dequantize_state_dict_int8 = mod.dequantize_state_dict_int8
eval_val = mod.eval_val

try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    HAS_ZSTD = False


def load_checkpoint(path: str):
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)

    blob = path.read_bytes()
    try:
        raw = zlib.decompress(blob)
    except zlib.error:
        if not HAS_ZSTD:
            raise RuntimeError("Checkpoint appears zstd-compressed but zstandard is unavailable.")
        raw = zstd.ZstdDecompressor().decompress(blob)

    obj = torch.load(io.BytesIO(raw), map_location="cpu")
    return dequantize_state_dict_int8(obj)


@torch.inference_mode()
def eval_val_context_ablation(
    args,
    model,
    device,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    keep_last_n: int,
    batch_seqs: int = 8,
):
    """
    Evaluate only positions that have at most keep_last_n visible tokens of context.
    We blank the earlier prefix with token id 0, and score only the final keep_last_n positions.
    """
    seq_len = args.train_seq_len
    if keep_last_n <= 0 or keep_last_n > seq_len:
        raise ValueError(f"keep_last_n must be in [1, {seq_len}]")

    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = 0.0
    token_count = 0
    byte_count = 0.0

    model.eval()

    for batch_seq_start in range(0, total_seqs, batch_seqs):
        batch_seq_end = min(batch_seq_start + batch_seqs, total_seqs)
        raw_start = batch_seq_start * seq_len
        raw_end = batch_seq_end * seq_len + 1

        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
        x_full = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)

        x_eval = x_full.clone()
        prefix_len = seq_len - keep_last_n
        if prefix_len > 0:
            x_eval[:, :prefix_len] = 0

        full_logits, _ = model._compute_logits(x_eval)
        logits = full_logits.float().view(-1, args.vocab_size)
        targets = y.reshape(-1)

        per_tok_nll = -F.log_softmax(logits * model.logit_sharpen, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)

        loss_mask = torch.zeros_like(y, dtype=torch.bool)
        loss_mask[:, prefix_len:] = True
        flat_mask = loss_mask.reshape(-1)

        loss_sum += per_tok_nll[flat_mask].sum().item()
        token_count += int(flat_mask.sum().item())

        prev_ids = x_full.reshape(-1)
        tgt_ids = y.reshape(-1)
        token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
        byte_count += token_bytes[flat_mask].sum().item()

    val_loss = loss_sum / token_count
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (token_count / byte_count)
    return val_loss, val_bpb


def main():
    ckpt_path = os.environ.get("CKPT", "final_model.int8.ptz")
    keep_list_env = os.environ.get("KEEP_LIST", "2,4,8,16,32,64,128")
    keep_list = [int(x) for x in keep_list_env.split(",") if x.strip()]

    args = Hyperparameters()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Tokenizer vocab {sp.vocab_size()} != VOCAB_SIZE {args.vocab_size}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    model = GPT(
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
    ).to(device).bfloat16()

    # Match train/eval setup from your script
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    model.bigram_prev_factors.data = model.bigram_prev_factors.data.float()
    model.bigram_next_factors.data = model.bigram_next_factors.data.float()

    state = load_checkpoint(ckpt_path)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"checkpoint={ckpt_path}")
    print(f"model_dim={args.model_dim} num_layers={args.num_layers} num_heads={args.num_heads} num_kv_heads={args.num_kv_heads}")
    print(f"mlp_mult={args.mlp_mult} rope_partial_dims={args.rope_partial_dims} encoder_layer_frac={args.encoder_layer_frac} bigram_rank={args.bigram_rank}")
    print(f"seq_len={args.train_seq_len}")
    print()

    # Full eval sanity check: should be near your final roundtrip number
    full_loss, full_bpb = eval_val(
        args=args,
        model=model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    print(f"keep=full  val_loss={full_loss:.8f}  val_bpb={full_bpb:.8f}")
    print()

    for n in keep_list:
        loss, bpb = eval_val_context_ablation(
            args=args,
            model=model,
            device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            keep_last_n=n,
        )
        print(
            f"keep={n:<4d} val_loss={loss:.8f}  val_bpb={bpb:.8f}  "
            f"delta_loss={loss - full_loss:+.8f}  delta_bpb={bpb - full_bpb:+.8f}"
        )


if __name__ == "__main__":
    main()