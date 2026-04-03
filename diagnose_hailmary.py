
from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


def load_module(module_path: str):
    module_path = str(Path(module_path).resolve())
    spec = importlib.util.spec_from_file_location("hailmary_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_state(module: Any, checkpoint_path: str):
    checkpoint_path = str(Path(checkpoint_path).resolve())
    if checkpoint_path.endswith(".ptz"):
        blob = Path(checkpoint_path).read_bytes()
        try:
            raw = __import__("zlib").decompress(blob)
        except Exception:
            try:
                import zstandard as zstd  # type: ignore
            except Exception as e:
                raise RuntimeError("Need zstandard installed to read .ptz compressed with zstd") from e
            raw = zstd.ZstdDecompressor().decompress(blob)
        obj = torch.load(io.BytesIO(raw), map_location="cpu")
        return module.dequantize_state_dict_int8(obj)
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def clone_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in sd.items()}


def zero_like_(state: dict[str, torch.Tensor], names: list[str]):
    for n in names:
        if n in state:
            state[n].zero_()


def set_scalar_(state: dict[str, torch.Tensor], name: str, value: float):
    if name in state:
        state[name].fill_(value)


def build_model(module: Any, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.GPT(
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
        ngram_num_hashes=args.ngram_num_hashes,
        ngram_hash_size=args.ngram_hash_size,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, module.CastedLinear):
            m.float()
    module.restore_low_dim_params_to_fp32(model)
    return model, device


def apply_variant(base_sd: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    sd = clone_state_dict(base_sd)

    if variant == "full":
        return sd

    if variant == "bigram_only":
        set_scalar_(sd, "ngram_scale", 0.0)
        set_scalar_(sd, "transformer_scale", 0.0)
        return sd

    if variant == "bigram_plus_transformer":
        set_scalar_(sd, "ngram_scale", 0.0)
        return sd

    if variant == "bigram_plus_ngram":
        set_scalar_(sd, "transformer_scale", 0.0)
        return sd

    if variant == "transformer_plus_ngram":
        zero_like_(sd, ["bigram_prev_factors", "bigram_next_factors"])
        return sd

    if variant == "transformer_only":
        zero_like_(sd, ["bigram_prev_factors", "bigram_next_factors"])
        set_scalar_(sd, "ngram_scale", 0.0)
        return sd

    if variant == "ngram_only":
        zero_like_(sd, ["bigram_prev_factors", "bigram_next_factors"])
        set_scalar_(sd, "transformer_scale", 0.0)
        return sd

    if variant == "no_bigram":
        zero_like_(sd, ["bigram_prev_factors", "bigram_next_factors"])
        return sd

    if variant == "no_ngram":
        set_scalar_(sd, "ngram_scale", 0.0)
        return sd

    if variant == "no_transformer":
        set_scalar_(sd, "transformer_scale", 0.0)
        return sd

    raise ValueError(f"Unknown variant: {variant}")


def summarize_contributions(results: dict[str, dict[str, float]]) -> dict[str, float]:
    full = results["full"]["val_bpb"]
    out = {}
    for k, v in results.items():
        if k == "full":
            continue
        out[f"delta_vs_full::{k}"] = v["val_bpb"] - full
    return out


def main():
    parser = argparse.ArgumentParser(description="Ablate hailmary components on validation.")
    parser.add_argument("--model-script", default="train_gpt_hailmary.py")
    parser.add_argument("--checkpoint", default="final_model.pt")
    parser.add_argument("--variants", nargs="*", default=[
        "full",
        "bigram_only",
        "bigram_plus_ngram",
        "bigram_plus_transformer",
        "transformer_plus_ngram",
        "transformer_only",
        "no_bigram",
        "no_ngram",
        "no_transformer",
    ])
    parser.add_argument("--json-out", default="")
    args_ns = parser.parse_args()

    module = load_module(args_ns.model_script)

    # Reuse script defaults / env resolution from the training script.
    hp = module.Hyperparameters()

    model, device = build_model(module, hp)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    val_tokens = module.load_validation_tokens(hp.val_files, hp.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = module.build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )

    base_sd = load_state(module, args_ns.checkpoint)

    results: dict[str, dict[str, float]] = {}
    for variant in args_ns.variants:
        sd = apply_variant(base_sd, variant)
        model.load_state_dict(sd, strict=False)
        model.eval()
        val_loss, val_bpb = module.eval_val(
            hp, model, rank=0, world_size=1, device=device, grad_accum_steps=8,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        results[variant] = {"val_loss": float(val_loss), "val_bpb": float(val_bpb)}
        print(f"{variant:>22}  val_loss={val_loss:.8f}  val_bpb={val_bpb:.8f}")

    deltas = summarize_contributions(results)
    print("\nDeltas vs full (positive means worse than full):")
    for k, v in deltas.items():
        print(f"{k:>30}  {v:+.8f}")

    if args_ns.json_out:
        payload = {"results": results, "deltas": deltas}
        Path(args_ns.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args_ns.json_out}")


if __name__ == "__main__":
    main()
