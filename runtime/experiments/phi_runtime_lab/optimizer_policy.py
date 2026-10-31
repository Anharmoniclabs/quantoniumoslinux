"""Opt-in weight-decay grouping for an isolated PhiState training candidate.

This does not change any runner's defaults, reset optimizer moments, increase
learning rates, or demonstrate a coherence fix. Apply AFTER loading an existing
optimizer state when creating a candidate. To resume that candidate, apply the
same grouping to a freshly constructed optimizer BEFORE loading its saved state.
"""
from __future__ import annotations

import torch

POLICY = "phistate-structured-no-decay-v1"


def protect_structured_parameters(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> dict:
    """Keep matrix weight decay; remove it from vectors, scalars and biases.

    In PhiState this protects recurrent decay/phase/readout parameters, RMSNorm
    scales, mixer gains and the two-dimensional attention-relative bias. Existing
    parameter identities, per-parameter state, group options and group-relative
    parameter order are preserved. No forward/backward/update is performed.

    Splitting groups changes the checkpoint's group layout. An old optimizer
    checkpoint must be loaded before this conversion; a converted checkpoint
    must be loaded into an optimizer converted by this function as well.
    """
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)

    groups, changes, seen = [], [], set()
    # Validate and prepare completely before modifying the optimizer.
    for original_index, group in enumerate(optimizer.param_groups):
        decay, protected = [], []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in aliases:
                raise ValueError("Optimizer contains a parameter outside the supplied model")
            if identity in seen:
                raise ValueError("Optimizer contains duplicate parameters")
            seen.add(identity)
            names = aliases[identity]
            exclude = parameter.ndim < 2 or any(
                name == "bias" or name.endswith(".bias") for name in names
            )
            (protected if exclude else decay).append(parameter)

        for parameters, excluded in ((decay, False), (protected, True)):
            if not parameters:
                continue
            new_group = dict(group)
            new_group["params"] = parameters
            # PyTorch supports optional param_names; keep it aligned after split.
            if "param_names" in group:
                if len(group["param_names"]) != len(group["params"]):
                    raise ValueError("Optimizer param_names do not match its parameters")
                names_by_id = dict(zip(map(id, group["params"]), group["param_names"]))
                new_group["param_names"] = [names_by_id[id(p)] for p in parameters]
            old_decay = group.get("weight_decay", optimizer.defaults.get("weight_decay", 0.0))
            new_group["weight_decay"] = 0.0 if excluded else old_decay
            groups.append(new_group)
            changes.append({
                "original_group": original_index,
                "excluded_from_weight_decay": excluded,
                "weight_decay_before": float(old_decay),
                "weight_decay_after": float(new_group["weight_decay"]),
                "parameters": [aliases[id(p)][0] for p in parameters],
            })

    # Do not reinstantiate the optimizer: its moments and counters stay attached
    # to the original Parameter objects. All other group options are copied.
    optimizer.param_groups[:] = groups
    return {"policy": POLICY, "groups": changes}
