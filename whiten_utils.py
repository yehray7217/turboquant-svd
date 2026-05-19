import os
import json
import re
import random
import torch
import sys
from tqdm import tqdm
import torch.nn as nn
import click
from pathlib import Path
from typing import Tuple
from modules.svd_linear import SVDLinear
#from modules.multilevel_svd_linear import MultiSVDLinear

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

def attach_nan_hooks_to_factorized_layers(model: nn.Module) -> None:
    """
    在所有 factorized linear（SVDLinear / MultiSVDLinear）以及一般 Linear / Norm 上掛 NaN 偵測 hook。
    一旦某層輸入或輸出有 NaN/Inf，就印出該層名稱與一些權重統計。
    """

    def _check_tensor(tag: str, name: str, module: nn.Module, t: torch.Tensor) -> None:
        if not torch.is_tensor(t):
            return
        if torch.isnan(t).any() or torch.isinf(t).any():
            # 基本資訊
            print(f"[NaN HOOK][{tag}] {name} ({type(module).__name__}) has NaN/Inf")
            try:
                with torch.no_grad():
                    if hasattr(module, "weight") and torch.is_tensor(module.weight):
                        w = module.weight
                        w_min = w.min().item()
                        w_max = w.max().item()
                        w_std = w.std().item()
                        print(
                            f"  [weight] min={w_min:.3e}, max={w_max:.3e}, std={w_std:.3e}, "
                            f"dtype={w.dtype}, device={w.device}"
                        )
            except Exception as e:
                print(f"  [NaN HOOK] failed to print weight stats: {e}")

    def make_hook(name: str):
        def hook(module: nn.Module, inputs, outputs):
            # 檢查輸入
            if isinstance(inputs, (tuple, list)):
                for idx, x in enumerate(inputs):
                    if torch.is_tensor(x):
                        _check_tensor(f"IN[{idx}]", name, module, x)
            elif torch.is_tensor(inputs):
                _check_tensor("IN", name, module, inputs)

            # 檢查輸出
            if torch.is_tensor(outputs):
                _check_tensor("OUT", name, module, outputs)
            elif isinstance(outputs, (tuple, list)):
                for idx, y in enumerate(outputs):
                    if torch.is_tensor(y):
                        _check_tensor(f"OUT[{idx}]", name, module, y)

        return hook

    # 盡量涵蓋：我們自己定義的 factorized linear + 普通 Linear + Norm 類
    hook_types = (SVDLinear, nn.Linear, nn.LayerNorm)

    # RMSNorm 在 LLaMA 裡通常不是標準 nn.Module 類名，可以用名字判斷
    def is_rmsnorm(m: nn.Module) -> bool:
        return "rmsnorm" in m.__class__.__name__.lower()

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, hook_types) or is_rmsnorm(module):
            module.register_forward_hook(make_hook(name))
            count += 1

    print(f"[NaN HOOK] registered on {count} modules.")

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def _whiten_monolithic_cache_file(model_id: str, calib_dataset: str) -> str:
    if calib_dataset == "wikitext2":
        dataset_tag = "w2"
    elif calib_dataset == "c4":
        dataset_tag = "c4"
    else:
        raise ValueError(f"Not supported calib_dataset: {calib_dataset}")
    return f"cache/whiten/{model_id.replace('/','_')}_{dataset_tag}_scaling_matrices_fp16.pt"


def _whiten_sharded_cache_dir(model_id: str, calib_dataset: str) -> Path:
    monolithic = Path(_whiten_monolithic_cache_file(model_id, calib_dataset))
    return monolithic.with_suffix("").with_name(monolithic.stem + "_layers")


def _whiten_layer_shard_path(cache_dir: Path, layer_idx: int) -> Path:
    return cache_dir / f"layer_{int(layer_idx):02d}.pt"


def _whiten_shard_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _write_whiten_shard_manifest(
    cache_dir: Path,
    *,
    model_id: str,
    calib_dataset: str,
    num_layers: int,
    complete: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "whiten_scaling_matrices_layer_shards_v1",
        "model_id": str(model_id),
        "calib_dataset": str(calib_dataset),
        "num_layers": int(num_layers),
        "complete": bool(complete),
        "layer_files": [
            _whiten_layer_shard_path(cache_dir, i).name
            for i in range(int(num_layers))
        ],
    }
    _whiten_shard_manifest_path(cache_dir).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _whiten_sharded_cache_is_complete(cache_dir: Path, num_layers: int) -> bool:
    manifest_path = _whiten_shard_manifest_path(cache_dir)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not bool(manifest.get("complete", False)):
        return False
    if int(manifest.get("num_layers", -1)) != int(num_layers):
        return False
    return all(
        _whiten_layer_shard_path(cache_dir, i).exists()
        for i in range(int(num_layers))
    )


def _attach_sharded_whiten_cache_metadata(model, cache_dir: Path, num_layers: int) -> None:
    model._whiten_sharded_cache_dir = str(cache_dir)
    model._whiten_sharded_num_layers = int(num_layers)


def _load_whiten_layer_shard(cache_dir: Path, layer_idx: int):
    path = _whiten_layer_shard_path(cache_dir, layer_idx)
    if not path.exists():
        raise FileNotFoundError(f"Missing whiten layer shard: {path}")
    return torch.load(path, map_location="cpu")


def _layer_idx_from_module_name(layername: str):
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", str(layername))
    if m is None:
        return None
    return int(m.group(1))


@torch.no_grad()
def profle_aat_large_scale(name, model, calib_loader, calib_dataset, dev):
    """Generate whitening scaling matrices as per-layer CPU-FP16 shards."""
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"Unsupported model family for whitening: {name}")

    cache_dir = _whiten_sharded_cache_dir(name, calib_dataset)
    num_layers = len(layers)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_whiten_shard_manifest(
        cache_dir,
        model_id=name,
        calib_dataset=calib_dataset,
        num_layers=num_layers,
        complete=False,
    )

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            if cache["position_ids"] is None:
                cache["position_ids"] = kwargs["position_ids"]
            else:
                cache["position_ids"] = torch.cat(
                    (cache["position_ids"], kwargs["position_ids"]),
                    dim=0,
                )
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = None
    position_ids = cache["position_ids"]

    for i in tqdm(range(num_layers)):
        shard_path = _whiten_layer_shard_path(cache_dir, i)
        layer = layers[i].to(dev)
        layer_dev = layer.self_attn.q_proj.weight.device

        if shard_path.exists():
            click.secho(
                f"[whiten] Layer shard already exists, reuse: {shard_path}",
                fg="yellow",
            )
            for j in range(inps.shape[0]):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids[j].unsqueeze(0).to(layer_dev),
                )[0]
            layers[i] = layer.cpu()
            inps = outs
            torch.cuda.empty_cache()
            continue

        subset = find_layers(layer)

        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1, 2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()

        handles = []
        for module_name in subset:
            subset[module_name].scaling_diag_matrix = 0
            handles.append(subset[module_name].register_forward_hook(hook))

        for j in range(inps.shape[0]):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids[j].unsqueeze(0).to(layer_dev),
            )[0]

        for h in handles:
            h.remove()

        layer = layer.cpu()
        for module_name in subset:
            subset[module_name].scaling_diag_matrix = (
                subset[module_name].scaling_diag_matrix.cpu()
            )
        torch.cuda.empty_cache()

        layer_scaling_matrices = {}
        for module_name in subset:
            print(module_name)
            raw_scaling_diag_matrix = (
                subset[module_name].scaling_diag_matrix.double().to(layer_dev)
            )
            try:
                scaling_diag_matrix = torch.linalg.cholesky(
                    raw_scaling_diag_matrix
                ).float()
            except Exception:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    print("Warning: raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    print("Warning: raw scaling_diag_matrix contains Inf!")
                if not torch.equal(
                    raw_scaling_diag_matrix, raw_scaling_diag_matrix.T
                ):
                    print("Warning: raw scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (
                    -eigenvalues[0] + 1e-3
                ) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(layer_dev)
                scaling_diag_matrix = torch.linalg.cholesky(
                    raw_scaling_diag_matrix
                ).float()
                if torch.isnan(scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains Inf!")
                del eigenvalues

            layer_scaling_matrices[module_name] = (
                scaling_diag_matrix.detach().half().cpu()
            )
            del raw_scaling_diag_matrix, scaling_diag_matrix
            torch.cuda.empty_cache()

        torch.save(layer_scaling_matrices, shard_path)
        click.secho(
            f"[whiten] Saved layer shard {i+1}/{num_layers}: {shard_path}",
            fg="cyan",
        )

        del layer_scaling_matrices
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()

    _write_whiten_shard_manifest(
        cache_dir,
        model_id=name,
        calib_dataset=calib_dataset,
        num_layers=num_layers,
        complete=True,
    )
    _attach_sharded_whiten_cache_metadata(model, cache_dir, num_layers)
    model.config.use_cache = use_cache
    click.secho(f"[whiten] Completed sharded cache: {cache_dir}", fg="green")
    return {"format": "layer_shards", "cache_dir": str(cache_dir), "num_layers": num_layers}

def whiten_model(name, model, dev):
    model.eval()
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    
    print("Start whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data.float().cuda()
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.float().cuda()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    print("Warning: scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            # scaling_matrix_invs[name] = scaling_matrix_inv
            W_scale = torch.matmul(W, scaling_diag_matrix)
            subset[name].weight = torch.nn.parameter.Parameter(W_scale)
            
            if torch.allclose(subset[name].weight.data, W):
                print("Warning: whitening failed!") 
            # subset[name].u, subset[name].s, subset[name].vt = torch.linalg.svd(W_scale, full_matrices=False)
            # subset[name].vt = torch.matmul(subset[name].vt, scaling_matrix_inv)
            # W_scale = scaling_matrix_inv = scaling_diag_matrix = raw_scaling_diag_matrix = None
            # del W_scale, scaling_matrix_inv, scaling_diag_matrix, raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()

    print("Finish whitening!")

# This function is used to insert the scaling diag matrix into each linear module
# The goal is same as calib_input_distribution in act_aware_utils.py
def insert_whiten_scale_matrix(model, calib_loader, calib_dataset="wikitext2", dev="cuda:0"):
    """Prepare whitening scale matrices with sharded-cache preference."""
    model_id = model.config._name_or_path
    cache_file = _whiten_monolithic_cache_file(model_id, calib_dataset)
    cache_dir = _whiten_sharded_cache_dir(model_id, calib_dataset)
    layers = model.model.layers
    num_layers = len(layers)

    click.secho(f"[whiten] Calibration dataset: {calib_dataset}", fg="yellow")

    if _whiten_sharded_cache_is_complete(cache_dir, num_layers):
        click.secho(
            f"[whiten] Load sharded scaling cache metadata: {cache_dir}",
            fg="yellow",
        )
        _attach_sharded_whiten_cache_metadata(model, cache_dir, num_layers)
        return

    if os.path.exists(cache_file):
        click.secho(
            f"[whiten] Legacy monolithic cache found: {cache_file}",
            fg="yellow",
        )
        scaling_matrics = torch.load(cache_file, map_location="cpu")
        for i in tqdm(range(num_layers)):
            layer = layers[i]
            subset = find_layers(layer)
            for module_name in subset:
                if module_name in scaling_matrics[i]:
                    subset[module_name].scaling_diag_matrix = (
                        scaling_matrics[i][module_name]
                    )
        return

    click.secho(f"[whiten] No complete sharded cache: {cache_dir}", fg="red")
    click.secho("[whiten] Generate whiten layer shards ...", fg="yellow")
    profle_aat_large_scale(
        model_id,
        model,
        calib_loader=calib_loader,
        calib_dataset=calib_dataset,
        dev=dev,
    )
    assert _whiten_sharded_cache_is_complete(cache_dir, num_layers), (
        f"Whiten sharded cache was not completed: {cache_dir}"
    )
    _attach_sharded_whiten_cache_metadata(model, cache_dir, num_layers)
    return

def whiten_decomposition(
    linear: nn.Linear,
    rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    w = linear.weight.data.float()
    H, W = w.size()

    try:
        scaling_diag_matrix = linear.scaling_diag_matrix.to(w.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")
    
    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(w.to(torch.float32), scaling_diag_matrix.to(torch.float32))
    
    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    
    V = torch.matmul(Vt, scaling_matrix_inv)
    
    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]

    # Check for nan
    if (S != S).any():
        print("nan in S")
        raise ValueError("nan in S")
   
    if (U != U).any():
        print("nan in U")
        raise ValueError("nan in U")
    
    if (V != V).any():
        print("nan in V")
        raise ValueError("nan in V")
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma)
    R = torch.matmul(sqrtSigma, V)

    # Log
    remain_param_ratio = ((H + W) * rank) / (H * W) * 100
    rank_ratio = rank / min(H, W) * 100
    print(
        f"Remaining Rank: {rank} ({rank_ratio:.2f}%) | Num Parameters: {(H + W) * rank} / {H * W} ({remain_param_ratio:.2f}%)")

    return L, R

def compress_model_whiten(model, selection_result, args):
    """Compress a whitening-prepared model with optional lazy shard loading."""
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}

    modules = [model]
    while modules:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sharded_cache_dir = getattr(model, "_whiten_sharded_cache_dir", None)
    shard_cache = None
    shard_layer_idx = None

    def ensure_shard_for_layer(layer_idx: int):
        nonlocal shard_cache, shard_layer_idx
        if sharded_cache_dir is None:
            return None
        if shard_layer_idx == layer_idx and shard_cache is not None:
            return shard_cache
        if shard_cache is not None:
            del shard_cache
            shard_cache = None
            torch.cuda.empty_cache()
        shard_layer_idx = int(layer_idx)
        shard_cache = _load_whiten_layer_shard(
            Path(sharded_cache_dir),
            shard_layer_idx,
        )
        click.secho(
            f"[whiten] Loaded compression shard layer {shard_layer_idx:02d}",
            fg="cyan",
        )
        return shard_cache

    for layername, rank_cfg in selection_result.items():
        print(layername, end=" ")

        if (
            layername == "lm_head"
            or layername.endswith(".lm_head")
            or layername.endswith("lm_head")
        ):
            print("[skip] lm_head", end=" ")
            continue

        if layername not in module_dict:
            print(f"[skip] missing module: {layername}", end=" ")
            continue

        raw_linear = module_dict[layername]
        if not isinstance(raw_linear, nn.Linear):
            print(f"[skip] not nn.Linear: {layername}", end=" ")
            continue

        if sharded_cache_dir is not None and not hasattr(
            raw_linear, "scaling_diag_matrix"
        ):
            layer_idx = _layer_idx_from_module_name(layername)
            if layer_idx is None:
                print(f"[skip] no layer idx for shard lookup: {layername}", end=" ")
                continue
            shard = ensure_shard_for_layer(layer_idx)
            short_name = layername.split(
                f"model.layers.{layer_idx}.",
                1,
            )[-1]
            if short_name not in shard:
                print(f"[skip] shard missing scaling matrix: {short_name}", end=" ")
                continue
            raw_linear.scaling_diag_matrix = shard[short_name]

        if not hasattr(raw_linear, "scaling_diag_matrix"):
            print(f"[skip] no scaling_diag_matrix: {layername}", end=" ")
            continue

        info = linear_info[raw_linear]

        if isinstance(rank_cfg, dict):
            if "rank" in rank_cfg:
                rank = int(rank_cfg["rank"])
            elif "outer" in rank_cfg:
                rank = int(rank_cfg["outer"])
            else:
                raise ValueError(
                    f"Unexpected rank_cfg dict for {layername}: {rank_cfg}"
                )
        else:
            rank = int(rank_cfg)

        if getattr(args, "search_with_succinct", False):
            svd_linear = SVDLinear.from_linear_whiten_rank(
                raw_linear,
                rank=rank,
                name=layername,
                succinct=True,
                sigma_fuse=getattr(args, "sigma_fuse", True),
            ).cpu()
        else:
            svd_linear = SVDLinear.from_linear_whiten_rank(
                raw_linear,
                name=layername,
                rank=rank,
            ).cpu()

        setattr(info["father"], info["name"], svd_linear)

        try:
            del raw_linear.scaling_diag_matrix
        except Exception:
            pass

    if shard_cache is not None:
        del shard_cache
    torch.cuda.empty_cache()

