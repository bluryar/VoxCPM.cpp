#!/usr/bin/env python3
"""
Convert VoxCPM model weights to GGUF format.

Usage:
    uv run python scripts/convert_voxcpm_to_gguf.py /path/to/VoxCPM1.5 --output voxcpm.gguf

Copyright 2025 OpenBMB
Licensed under the Apache License, Version 2.0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file

# Add parent directory to path for gguf import
sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf
from gguf import GGUFReader

REPO_ROOT = Path(__file__).parent.parent.parent
VOXCPM_SRC_ROOT = REPO_ROOT / "vendor" / "VoxCPM" / "src"
if str(VOXCPM_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(VOXCPM_SRC_ROOT))

from voxcpm.modules.audiovae.audio_vae import AudioVAEConfig


# VoxCPM uses custom architecture, but we use LLAMA as base for GGUF
# since the transformer structure is similar


def get_tensor_dtype(tensor: np.ndarray) -> gguf.GGMLQuantizationType:
    """Get GGML dtype from numpy array."""
    if tensor.dtype == np.float32:
        return gguf.GGMLQuantizationType.F32
    elif tensor.dtype == np.float16:
        return gguf.GGMLQuantizationType.F16
    elif tensor.dtype == np.int32:
        return gguf.GGMLQuantizationType.I32
    elif tensor.dtype == np.int64:
        return gguf.GGMLQuantizationType.I64
    else:
        return gguf.GGMLQuantizationType.F32


# NOTE: Conv1d weights do NOT need to be transposed!
#
# GGML conv_1d kernel layout: ne0*ne1 = IC*K, ne2 = OC
# GGUF stores numpy shape reversed: numpy (OC, IC, K) -> GGUF [K, IC, OC] -> GGML ne [K, IC, OC]
# This satisfies ne0*ne1 = K*IC, ne2 = OC, which is correct!
#
# So PyTorch Conv1d weights [OC, IC, K] can be written directly to GGUF without any transpose.


def process_audiovae_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Process AudioVAE weights for ggbond GGML format.

    Conv1d weights do NOT need transposition - PyTorch [OC, IC, K] layout is correct.

    Snake alpha parameters need shape transformation for correct GGML broadcasting:
    - PyTorch Snake1d alpha: [1, C, 1]
    - GGUF stores numpy (1, C) -> GGML ne [C, 1] (WRONG)
    - We need GGML ne [1, C] for broadcasting with x [T, C]
    - So we store numpy (C, 1) -> GGML ne [1, C] (CORRECT)

    Args:
        weights: Dict of tensor name -> numpy array (with audio_vae. prefix)

    Returns:
        Processed weights dict with alpha shapes fixed for GGML broadcast
    """
    processed = {}
    for name, tensor in weights.items():
        # Handle Snake alpha parameters - fix shape for GGML broadcast
        if 'alpha' in name:
            # PyTorch Snake1d alpha: [1, C, 1] -> squeeze to [C]
            # For GGML broadcast with x [T, C], alpha needs GGML shape [1, C]
            # numpy [C, 1] -> GGML [1, C] (reversed)
            if tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[2] == 1:
                # [1, C, 1] -> [C] -> [C, 1]
                tensor = tensor.squeeze()[:, np.newaxis]
            elif tensor.ndim == 2 and tensor.shape[0] == 1:
                # Already [1, C] -> change to [C, 1]
                tensor = tensor.T
        processed[name] = tensor
    return processed


def convert_weight_norm(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert weight_norm separated weights (weight_g, weight_v) back to single weight.

    PyTorch's weight_norm stores weights as:
        weight = weight_g * weight_v / ||weight_v||_2

    For dim=0 (default), the norm is computed over all dimensions EXCEPT dim 0.
    weight_g has shape [C_out, 1, 1] for Conv1d with dim=0.

    Args:
        state_dict: Original state dict with weight_g and weight_v

    Returns:
        New state dict with merged weights
    """
    new_state_dict = {}
    weight_norm_keys = set()

    # Find all weight_norm keys
    for key in state_dict:
        if key.endswith('.weight_g') or key.endswith('.weight_v'):
            weight_norm_keys.add(key.rsplit('.', 1)[0])

    # Process non-weight_norm keys (including biases)
    for key, value in state_dict.items():
        # Only skip weight_g and weight_v themselves, not biases
        if key.endswith('.weight_g') or key.endswith('.weight_v'):
            continue  # Skip weight_norm keys, process later
        else:
            new_state_dict[key] = value

    # Merge weight_norm weights
    for base_key in weight_norm_keys:
        weight_g = state_dict[f"{base_key}.weight_g"]
        weight_v = state_dict[f"{base_key}.weight_v"]

        # Infer the normalized dim from weight_g shape
        # For dim=0: weight_g has shape [C_out, 1, 1] for Conv1d
        # Find which dims are size 1 in weight_g but not in weight_v
        weight_v_ndim = weight_v.dim()
        norm_dims = []
        for d in range(weight_v_ndim):
            if weight_g.shape[d] == 1 and weight_v.shape[d] > 1:
                norm_dims.append(d)

        # Compute norm over all dimensions except the normalized dim
        if norm_dims:
            norm = weight_v.norm(dim=tuple(norm_dims), keepdim=True)
        else:
            # Fallback: norm over all dimensions
            norm = weight_v.norm(keepdim=True)

        weight = weight_g * weight_v / (norm + 1e-12)

        new_state_dict[f"{base_key}.weight"] = weight

    return new_state_dict


def infer_model_name(model_path: Path, config: Dict[str, Any]) -> str:
    """Infer a human-friendly model name for GGUF metadata."""
    for key in ("model_name", "name"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    path_name = model_path.name.lower()
    if "0.5b" in path_name:
        return "VoxCPM-0.5B"
    if "1.5" in path_name:
        return "VoxCPM-1.5"
    return "VoxCPM"


def unwrap_state_dict(state_dict_or_checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract a flat state_dict from common checkpoint layouts."""
    if isinstance(state_dict_or_checkpoint, dict):
        if "state_dict" in state_dict_or_checkpoint and isinstance(state_dict_or_checkpoint["state_dict"], dict):
            return state_dict_or_checkpoint["state_dict"]
        return state_dict_or_checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(state_dict_or_checkpoint)!r}")


def conv_stride_from_weight(weights: Dict[str, torch.Tensor], key: str) -> Optional[int]:
    tensor = weights.get(key)
    if tensor is None:
        return None
    kernel = int(tensor.shape[-1])
    if kernel % 2 != 0:
        return None
    return kernel // 2


def resolve_audio_vae_config(config: Dict[str, Any], vae_weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Return a complete AudioVAE config for GGUF metadata emission."""
    if "audio_vae_config" in config and isinstance(config["audio_vae_config"], dict):
        return dict(config["audio_vae_config"])

    default_cfg_obj = AudioVAEConfig()
    if hasattr(default_cfg_obj, "model_dump"):
        default_cfg = default_cfg_obj.model_dump()
    else:
        default_cfg = default_cfg_obj.dict()
    resolved = dict(default_cfg)

    encoder_bias = vae_weights.get("encoder.block.0.bias")
    if encoder_bias is not None:
        resolved["encoder_dim"] = int(encoder_bias.numel())

    latent_bias = vae_weights.get("encoder.fc_mu.bias")
    if latent_bias is not None:
        resolved["latent_dim"] = int(latent_bias.numel())

    decoder_bias = vae_weights.get("decoder.model.1.bias")
    if decoder_bias is not None:
        resolved["decoder_dim"] = int(decoder_bias.numel())

    encoder_rates: List[int] = []
    decoder_rates: List[int] = []
    for i in range(1, 16):
        stride = conv_stride_from_weight(vae_weights, f"encoder.block.{i}.block.4.weight")
        if stride is None:
            break
        encoder_rates.append(stride)
    for i in range(2, 32):
        stride = conv_stride_from_weight(vae_weights, f"decoder.model.{i}.block.1.weight")
        if stride is None:
            break
        decoder_rates.append(stride)

    if encoder_rates:
        resolved["encoder_rates"] = encoder_rates
    if decoder_rates:
        resolved["decoder_rates"] = decoder_rates

    return resolved


def load_voxcpm_weights(model_path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load VoxCPM model weights from safetensors and audiovae.pth.

    Args:
        model_path: Path to VoxCPM model directory

    Returns:
        Tuple of (weights dict, config dict)
    """
    model_path = Path(model_path)

    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load main model weights
    safetensors_path = model_path / "model.safetensors"
    pytorch_bin_path = model_path / "pytorch_model.bin"
    if safetensors_path.exists():
        print(f"Loading main model from {safetensors_path}...")
        main_weights = load_file(safetensors_path)
    elif pytorch_bin_path.exists():
        print(f"Loading main model from {pytorch_bin_path}...")
        main_weights = unwrap_state_dict(torch.load(pytorch_bin_path, map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {model_path}")

    # Load AudioVAE weights
    audiovae_path = model_path / "audiovae.pth"
    if not audiovae_path.exists():
        raise FileNotFoundError(f"audiovae.pth not found in {model_path}")

    print(f"Loading AudioVAE from {audiovae_path}...")
    vae_checkpoint = torch.load(audiovae_path, map_location="cpu", weights_only=True)
    vae_weights = unwrap_state_dict(vae_checkpoint)

    # Convert weight_norm weights in AudioVAE
    print("Converting weight_norm weights in AudioVAE...")
    vae_weights = convert_weight_norm(vae_weights)
    config["audio_vae_config"] = resolve_audio_vae_config(config, vae_weights)

    # Add audio_vae prefix to VAE weights
    for key, value in vae_weights.items():
        main_weights[f"audio_vae.{key}"] = value

    # Convert all weights to numpy
    weights = {}
    for key, value in main_weights.items():
        # Convert bfloat16 to float32 first (numpy doesn't support bfloat16 directly)
        if value.dtype == torch.bfloat16:
            value = value.float()
        tensor = value.numpy()
        weights[key] = tensor

    print(f"Loaded {len(weights)} tensors")

    # Process AudioVAE weights (transpose conv1d weights)
    print("Processing AudioVAE conv1d weights for GGML format...")
    weights = process_audiovae_weights(weights)

    return weights, config


def map_tensor_name(name: str) -> str:
    """
    Map PyTorch tensor name to GGUF tensor name.

    GGUF naming conventions:
        - token_embd.weight for embedding
        - blk.{i}.attn_q.weight for attention Q projection
        - blk.{i}.attn_k.weight for attention K projection
        - blk.{i}.attn_v.weight for attention V projection
        - blk.{i}.attn_output.weight for attention O projection
        - blk.{i}.ffn_gate.weight for SwiGLU gate projection
        - blk.{i}.ffn_up.weight for SwiGLU up projection
        - blk.{i}.ffn_down.weight for SwiGLU down projection
        - blk.{i}.attn_norm.weight for input RMSNorm
        - blk.{i}.ffn_norm.weight for post-attention RMSNorm
    """
    # TSLM (base_lm) mappings
    if name.startswith("base_lm."):
        name = name[len("base_lm."):]

        # Embedding
        if name == "embed_tokens.weight":
            return "token_embd.weight"

        # Final norm
        if name == "norm.weight":
            return "output_norm.weight"

        # Layer mappings
        if name.startswith("layers."):
            parts = name.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])

            if rest == "input_layernorm.weight":
                return f"blk.{layer_idx}.attn_norm.weight"
            if rest == "post_attention_layernorm.weight":
                return f"blk.{layer_idx}.ffn_norm.weight"
            if rest == "self_attn.q_proj.weight":
                return f"blk.{layer_idx}.attn_q.weight"
            if rest == "self_attn.k_proj.weight":
                return f"blk.{layer_idx}.attn_k.weight"
            if rest == "self_attn.v_proj.weight":
                return f"blk.{layer_idx}.attn_v.weight"
            if rest == "self_attn.o_proj.weight":
                return f"blk.{layer_idx}.attn_output.weight"
            if rest == "mlp.gate_proj.weight":
                return f"blk.{layer_idx}.ffn_gate.weight"
            if rest == "mlp.up_proj.weight":
                return f"blk.{layer_idx}.ffn_up.weight"
            if rest == "mlp.down_proj.weight":
                return f"blk.{layer_idx}.ffn_down.weight"

    # RALM (residual_lm) mappings
    if name.startswith("residual_lm."):
        name = name[len("residual_lm."):]

        # Final norm
        if name == "norm.weight":
            return "residual_lm.output_norm.weight"

        # Layer mappings
        if name.startswith("layers."):
            parts = name.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])

            if rest == "input_layernorm.weight":
                return f"residual_lm.blk.{layer_idx}.attn_norm.weight"
            if rest == "post_attention_layernorm.weight":
                return f"residual_lm.blk.{layer_idx}.ffn_norm.weight"
            if rest == "self_attn.q_proj.weight":
                return f"residual_lm.blk.{layer_idx}.attn_q.weight"
            if rest == "self_attn.k_proj.weight":
                return f"residual_lm.blk.{layer_idx}.attn_k.weight"
            if rest == "self_attn.v_proj.weight":
                return f"residual_lm.blk.{layer_idx}.attn_v.weight"
            if rest == "self_attn.o_proj.weight":
                return f"residual_lm.blk.{layer_idx}.attn_output.weight"
            if rest == "mlp.gate_proj.weight":
                return f"residual_lm.blk.{layer_idx}.ffn_gate.weight"
            if rest == "mlp.up_proj.weight":
                return f"residual_lm.blk.{layer_idx}.ffn_up.weight"
            if rest == "mlp.down_proj.weight":
                return f"residual_lm.blk.{layer_idx}.ffn_down.weight"

    # LocEnc (feat_encoder) mappings
    if name.startswith("feat_encoder."):
        name = name[len("feat_encoder."):]

        if name == "in_proj.weight":
            return "locenc.in_proj.weight"
        if name == "in_proj.bias":
            return "locenc.in_proj.bias"
        if name == "special_token":
            return "locenc.special_token"

        # Encoder norm
        if name == "encoder.norm.weight":
            return "locenc.output_norm.weight"

        # Layer mappings
        if name.startswith("encoder.layers."):
            parts = name.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])

            if rest == "input_layernorm.weight":
                return f"locenc.blk.{layer_idx}.attn_norm.weight"
            if rest == "post_attention_layernorm.weight":
                return f"locenc.blk.{layer_idx}.ffn_norm.weight"
            if rest == "self_attn.q_proj.weight":
                return f"locenc.blk.{layer_idx}.attn_q.weight"
            if rest == "self_attn.k_proj.weight":
                return f"locenc.blk.{layer_idx}.attn_k.weight"
            if rest == "self_attn.v_proj.weight":
                return f"locenc.blk.{layer_idx}.attn_v.weight"
            if rest == "self_attn.o_proj.weight":
                return f"locenc.blk.{layer_idx}.attn_output.weight"
            if rest == "mlp.gate_proj.weight":
                return f"locenc.blk.{layer_idx}.ffn_gate.weight"
            if rest == "mlp.up_proj.weight":
                return f"locenc.blk.{layer_idx}.ffn_up.weight"
            if rest == "mlp.down_proj.weight":
                return f"locenc.blk.{layer_idx}.ffn_down.weight"

    # LocDiT (feat_decoder.estimator) mappings
    if name.startswith("feat_decoder.estimator."):
        name = name[len("feat_decoder.estimator."):]

        if name == "in_proj.weight":
            return "locdit.in_proj.weight"
        if name == "in_proj.bias":
            return "locdit.in_proj.bias"
        if name == "cond_proj.weight":
            return "locdit.cond_proj.weight"
        if name == "cond_proj.bias":
            return "locdit.cond_proj.bias"
        if name == "out_proj.weight":
            return "locdit.out_proj.weight"
        if name == "out_proj.bias":
            return "locdit.out_proj.bias"

        # Time MLP
        if name.startswith("time_mlp."):
            return f"locdit.{name}"
        if name.startswith("delta_time_mlp."):
            return f"locdit.{name}"

        # Decoder norm
        if name == "decoder.norm.weight":
            return "locdit.output_norm.weight"

        # Layer mappings
        if name.startswith("decoder.layers."):
            parts = name.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])

            if rest == "input_layernorm.weight":
                return f"locdit.blk.{layer_idx}.attn_norm.weight"
            if rest == "post_attention_layernorm.weight":
                return f"locdit.blk.{layer_idx}.ffn_norm.weight"
            if rest == "self_attn.q_proj.weight":
                return f"locdit.blk.{layer_idx}.attn_q.weight"
            if rest == "self_attn.k_proj.weight":
                return f"locdit.blk.{layer_idx}.attn_k.weight"
            if rest == "self_attn.v_proj.weight":
                return f"locdit.blk.{layer_idx}.attn_v.weight"
            if rest == "self_attn.o_proj.weight":
                return f"locdit.blk.{layer_idx}.attn_output.weight"
            if rest == "mlp.gate_proj.weight":
                return f"locdit.blk.{layer_idx}.ffn_gate.weight"
            if rest == "mlp.up_proj.weight":
                return f"locdit.blk.{layer_idx}.ffn_up.weight"
            if rest == "mlp.down_proj.weight":
                return f"locdit.blk.{layer_idx}.ffn_down.weight"

    # FSQ mappings
    if name.startswith("fsq_layer."):
        return f"fsq.{name[len('fsq_layer.'):]}"

    # Projection layers
    if name == "enc_to_lm_proj.weight":
        return "proj.enc_to_lm.weight"
    if name == "enc_to_lm_proj.bias":
        return "proj.enc_to_lm.bias"
    if name == "lm_to_dit_proj.weight":
        return "proj.lm_to_dit.weight"
    if name == "lm_to_dit_proj.bias":
        return "proj.lm_to_dit.bias"
    if name == "res_to_dit_proj.weight":
        return "proj.res_to_dit.weight"
    if name == "res_to_dit_proj.bias":
        return "proj.res_to_dit.bias"

    # Stop predictor
    if name.startswith("stop_"):
        return f"stop.{name}"

    # AudioVAE mappings
    if name.startswith("audio_vae."):
        return name  # Keep as-is for now

    # Default: keep original name
    return name


def flatten_config(config: Dict, prefix: str = "voxcpm") -> Dict[str, Any]:
    """
    Recursively flatten nested config dict into flat key-value pairs.

    Args:
        config: Nested config dict
        prefix: Key prefix (default: "voxcpm")

    Returns:
        Dict of flat key -> value pairs

    Example:
        {"lm_config": {"hidden_size": 1024}} -> {"voxcpm_lm_config_hidden_size": 1024}
    """
    result = {}

    for key, value in config.items():
        # Skip None values
        if value is None:
            continue

        flat_key = f"{prefix}_{key}"

        if isinstance(value, dict):
            # Recursively flatten nested dicts
            result.update(flatten_config(value, flat_key))
        elif isinstance(value, (list, tuple)):
            # Store arrays directly
            result[flat_key] = list(value)
        elif isinstance(value, bool):
            # Convert bool to int for GGUF
            result[flat_key] = 1 if value else 0
        elif isinstance(value, (int, float, str)):
            # Store primitive types directly
            result[flat_key] = value
        else:
            # Skip unsupported types
            pass

    return result


def write_gguf(
    output_path: str,
    weights: Dict[str, np.ndarray],
    config: Dict,
    model_name: str,
    tokenizer_info: Optional[Dict] = None,
):
    """
    Write model weights and config to GGUF file.

    Args:
        output_path: Output GGUF file path
        weights: Dict of tensor name -> numpy array
        config: Model config dict
        tokenizer_info: Dict with tokenizer info (tokens, toktypes, tokpre, merges, bos/eos/unk ids)
    """
    print(f"Writing GGUF to {output_path}...")

    # Create GGUF writer
    # Use "llama" as base architecture since VoxCPM shares similar transformer structure
    gguf_writer = gguf.GGUFWriter(
        path=output_path,
        arch="llama",
    )

    # Add model metadata - require all keys to exist (no defaults)
    lm_config = config["lm_config"]

    # General metadata
    gguf_writer.add_name(model_name)
    gguf_writer.add_architecture()
    gguf_writer.add_file_type(gguf.GGMLQuantizationType.F32)

    # LM config - direct access, fail if missing
    gguf_writer.add_context_length(lm_config["max_position_embeddings"])
    gguf_writer.add_embedding_length(lm_config["hidden_size"])
    gguf_writer.add_block_count(lm_config["num_hidden_layers"])
    gguf_writer.add_feed_forward_length(lm_config["intermediate_size"])
    gguf_writer.add_head_count(lm_config["num_attention_heads"])
    gguf_writer.add_head_count_kv(lm_config["num_key_value_heads"])
    gguf_writer.add_layer_norm_rms_eps(lm_config["rms_norm_eps"])
    gguf_writer.add_rope_freq_base(lm_config["rope_theta"])

    # Vocabulary
    vocab_size = lm_config["vocab_size"]
    gguf_writer.add_vocab_size(vocab_size)

    # Tokenizer
    if tokenizer_info:
        tokens = tokenizer_info.get("tokens", [])
        toktypes = tokenizer_info.get("toktypes", [])
        tokpre = tokenizer_info.get("tokpre", "default")
        merges = tokenizer_info.get("merges", [])
        bos_token_id = tokenizer_info.get("bos_token_id", 0)
        eos_token_id = tokenizer_info.get("eos_token_id", 0)
        unk_token_id = tokenizer_info.get("unk_token_id", 0)

        gguf_writer.add_tokenizer_model("gpt2")
        gguf_writer.add_tokenizer_pre(tokpre)
        gguf_writer.add_token_list(tokens)
        gguf_writer.add_token_types(toktypes)
        gguf_writer.add_token_merges(merges)
        gguf_writer.add_bos_token_id(bos_token_id)
        gguf_writer.add_eos_token_id(eos_token_id)
        gguf_writer.add_unk_token_id(unk_token_id)

    # Flatten and write all config fields with voxcpm_ prefix
    # This ensures all nested config values are written to GGUF
    flat_config = flatten_config(config, "voxcpm")

    # Also flatten lm_config with explicit prefix for C++ code access
    lm_flat = flatten_config(lm_config, "voxcpm_lm_config")
    flat_config.update(lm_flat)

    # Flatten encoder_config
    if "encoder_config" in config:
        encoder_flat = flatten_config(config["encoder_config"], "voxcpm_encoder_config")
        flat_config.update(encoder_flat)

    # Flatten dit_config
    if "dit_config" in config:
        dit_flat = flatten_config(config["dit_config"], "voxcpm_dit_config")
        flat_config.update(dit_flat)

    # Flatten audio_vae_config
    if "audio_vae_config" in config:
        vae_flat = flatten_config(config["audio_vae_config"], "voxcpm_audio_vae_config")
        flat_config.update(vae_flat)

    # Write flattened config to GGUF
    for key, value in sorted(flat_config.items()):
        # Skip fields that are already written via llama.* standard keys
        # or handled specially below
        skip_keys = {
        }
        if key in skip_keys:
            continue

        # Write based on type
        if isinstance(value, bool):
            gguf_writer.add_uint32(key, 1 if value else 0)
        elif isinstance(value, int):
            gguf_writer.add_uint32(key, value)
        elif isinstance(value, float):
            gguf_writer.add_float32(key, value)
        elif isinstance(value, str):
            gguf_writer.add_string(key, value)
        elif isinstance(value, list):
            # Determine array element type
            if not value:
                continue
            if isinstance(value[0], int):
                gguf_writer.add_array(key, [int(v) for v in value])
            elif isinstance(value[0], float):
                gguf_writer.add_array(key, [float(v) for v in value])
            elif isinstance(value[0], str):
                gguf_writer.add_array(key, [str(v) for v in value])

    # Add tensors
    print("Adding tensors...")
    tensor_count = 0
    for name, tensor in weights.items():
        gguf_name = map_tensor_name(name)

        # Get data type
        dtype = get_tensor_dtype(tensor)

        # Add tensor
        gguf_writer.add_tensor(
            name=gguf_name,
            tensor=tensor,
            raw_dtype=dtype,
        )
        tensor_count += 1

        if tensor_count % 100 == 0:
            print(f"  Added {tensor_count}/{len(weights)} tensors...")

    print(f"  Total: {tensor_count} tensors")

    # Write to file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print(f"GGUF file written to {output_path}")


def extract_gguf_info(gguf_path: str) -> Dict[str, Any]:
    """
    Read GGUF file and extract metadata, tokenizer, and weights information.

    Args:
        gguf_path: Path to the GGUF file

    Returns:
        Dict containing metadata, tokenizer, and weights info
    """
    print(f"\nReading GGUF file for verification: {gguf_path}...")
    reader = GGUFReader(gguf_path)

    # Extract metadata (all KV pairs)
    metadata = {}
    for key, field in reader.fields.items():
        try:
            value = field.contents()
            # Convert numpy types to Python native types for JSON serialization
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                value = value.item()
            metadata[key] = value
        except Exception as e:
            metadata[key] = f"<error: {e}>"

    # Extract tokenizer-specific info
    tokenizer_info = {
        "model": metadata.get("tokenizer.ggml.model"),
        "pre": metadata.get("tokenizer.ggml.pre"),
        "tokens_count": len(metadata.get("tokenizer.ggml.tokens", [])),
        "merges_count": len(metadata.get("tokenizer.ggml.merges", [])),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "unk_token_id": metadata.get("tokenizer.ggml.unk_token_id"),
        "token_types": [str(t) for t in metadata.get("tokenizer.ggml.token_type", [])][:10],  # First 10 for preview
        "first_5_merges": metadata.get("tokenizer.ggml.merges", [])[:5],
    }

    # Extract weights/tensor info
    weights_info = []
    total_params = 0
    total_bytes = 0

    for tensor in reader.tensors:
        tensor_info = {
            "name": tensor.name,
            "shape": tensor.shape.tolist(),
            "dtype": str(tensor.tensor_type),
            "n_elements": tensor.n_elements,
            "n_bytes": tensor.n_bytes,
        }
        weights_info.append(tensor_info)
        total_params += tensor.n_elements
        total_bytes += tensor.n_bytes

    # Create summary info
    result = {
        "file_info": {
            "path": gguf_path,
            "size_mb": os.path.getsize(gguf_path) / 1024 / 1024,
            "gguf_version": metadata.get("GGUF.version"),
            "tensor_count": len(reader.tensors),
            "kv_count": len(reader.fields),
        },
        "model_info": {
            "name": metadata.get("general.name"),
            "architecture": metadata.get("general.architecture"),
            "quantization_type": str(metadata.get("general.file_type")),
            "total_parameters": total_params,
            "total_parameters_billions": round(total_params / 1e9, 3),
            "total_bytes": total_bytes,
            "total_size_mb": total_bytes / 1024 / 1024,
        },
        "metadata": metadata,
        "tokenizer": tokenizer_info,
        "weights_summary": {
            "total_tensors": len(weights_info),
            "tensors": weights_info,
        },
    }

    return result


def write_gguf_info_json(gguf_path: str, output_dir: Optional[str] = None) -> str:
    """
    Read GGUF file and write info to JSON file.

    Args:
        gguf_path: Path to the GGUF file
        output_dir: Output directory for JSON file (default: same as GGUF file)

    Returns:
        Path to the output JSON file
    """
    # Extract info
    info = extract_gguf_info(gguf_path)

    # Determine output path
    gguf_path = Path(gguf_path)
    if output_dir is None:
        output_dir = gguf_path.parent
    else:
        output_dir = Path(output_dir)

    json_path = output_dir / f"{gguf_path.stem}_info.json"

    # Write JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    print(f"GGUF info written to {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("GGUF File Summary")
    print("=" * 60)
    print(f"File: {info['file_info']['path']}")
    print(f"Size: {info['file_info']['size_mb']:.2f} MB")
    print(f"GGUF Version: {info['file_info']['gguf_version']}")
    print(f"Tensors: {info['file_info']['tensor_count']}")
    print(f"KV Pairs: {info['file_info']['kv_count']}")
    print(f"\nModel: {info['model_info']['name']}")
    print(f"Architecture: {info['model_info']['architecture']}")
    print(f"Parameters: {info['model_info']['total_parameters_billions']}B")
    print(f"\nTokenizer: {info['tokenizer']['model']}")
    print(f"Tokens: {info['tokenizer']['tokens_count']}")
    print(f"Merges: {info['tokenizer']['merges_count']}")
    print(f"BOS ID: {info['tokenizer']['bos_token_id']}")
    print(f"EOS ID: {info['tokenizer']['eos_token_id']}")
    print("=" * 60)

    return str(json_path)


def main():
    parser = argparse.ArgumentParser(description="Convert VoxCPM model to GGUF format")
    parser.add_argument("model_path", type=str, help="Path to VoxCPM model directory")
    parser.add_argument("--output", "-o", type=str, default="voxcpm.gguf",
                        help="Output GGUF file path")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (default: model_path)")

    args = parser.parse_args()

    # Validate model path
    if not os.path.isdir(args.model_path):
        print(f"Error: {args.model_path} is not a directory")
        sys.exit(1)

    # Load weights
    weights, config = load_voxcpm_weights(args.model_path)
    model_name = infer_model_name(Path(args.model_path), config)

    # Load tokenizer
    tokenizer_path = args.tokenizer if args.tokenizer else args.model_path
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    tokenizer_json_path = Path(tokenizer_path)
    if tokenizer_json_path.is_dir():
        tokenizer_json_path = tokenizer_json_path / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_json_path}")

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    tokenizer_model = tokenizer_json.get("model", {})
    if tokenizer_model.get("type") != "BPE":
        raise ValueError(f"Expected BPE tokenizer, got {tokenizer_model.get('type')!r}")

    merges = tokenizer_model.get("merges", [])
    if not merges:
        raise ValueError("Tokenizer merges are empty; refusing to write incomplete GGUF metadata")

    vocab_size = config.get("lm_config", {}).get("vocab_size", len(tokenizer.vocab))
    reverse_vocab = {id_: tok for tok, id_ in tokenizer.vocab.items()}
    added_vocab = tokenizer.get_added_vocab()

    tokens = []
    toktypes = []
    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
        else:
            tok = reverse_vocab[i]
            if tok in added_vocab:
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
            tokens.append(tok)

    tokpre = "default"

    if len(tokens) != vocab_size:
        raise ValueError(f"Token count mismatch: expected vocab_size={vocab_size}, got {len(tokens)}")

    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    tokenizer_info = {
        "tokens": tokens,
        "toktypes": toktypes,
        "tokpre": tokpre,
        "merges": merges,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "unk_token_id": unk_token_id,
    }

    # Write GGUF
    write_gguf(args.output, weights, config, model_name, tokenizer_info)

    # Print file size
    file_size = os.path.getsize(args.output)
    print(f"Output file size: {file_size / 1024 / 1024:.2f} MB")

    # Read back and write info to JSON
    write_gguf_info_json(args.output)


if __name__ == "__main__":
    main()
