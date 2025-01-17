# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

NO_LORA = -100
LORA_BLOCK_MAPPING = []
if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class MeloConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})

    # melo_modified
    grace_layer: str = field(
        default=None,
        metadata={
            "help": "Module name as a grace layer"
        }
    )

    grace_config: dict = field(
        default=None,
        metadata={
            "help": "Default settings of the grace layer"
        }
    )

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.MELO


class MeloModel(torch.nn.Module):
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])
        self.add_grace(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def add_grace(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_melo_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace_grace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _check_grace_layer_exists(self, grace_layer: str, key):
        target_module_found = re.fullmatch(grace_layer, key)

        return target_module_found

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "num_rank_per_block": lora_config.grace_config['num_rank_per_block']
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _create_new_grace_module(self, config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": config.fan_in_fan_out,
        }

        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = config.fan_in_fan_out = False
        else:
            raise ValueError(
                f"Target grace module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` are supported."
            )
        new_module = GraceLinear(adapter_name, in_features, out_features, config.grace_config, bias=bias, **kwargs)

        return new_module

    def _find_and_replace_grace(self, adapter_name):
        config = self.peft_config[adapter_name]
        is_target_module_in_base_model = False
        grace_layer = config.grace_layer
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_grace_layer_exists(grace_layer, key):
                continue
            print(f"Target Grace Layer is found: {key}")
            is_target_module_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                raise ValueError("Cannot set LoraLayer as GraceLayer")
            new_module = self._create_new_grace_module(config, adapter_name, target)
            self._replace_module(parent, target_name, new_module, target)
        if not is_target_module_in_base_model:
            raise ValueError(
                f"Target grace modules {config.model.grace_layer} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]

        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                    lora_config.grace_config['num_rank_per_block']
                )
            elif isinstance(target, GraceLayer):
                raise ValueError("Cannot set GraceLayer as LoraLayer")
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def disable_grace_layer(self):
        for module in self.model.modules():
            if isinstance(module, GraceLayer):
                module.disable_grace = True

    def enable_grace_layer(self):
        for module in self.model.modules():
            if isinstance(module, GraceLayer):
                module.disable_grace = False

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    @staticmethod
    def _prepare_melo_config(peft_config, model_config):
        if peft_config.grace_layer is None:
            raise ValueError("Please specify `grace_layer` in `peft_config`")
        return peft_config

    @staticmethod
    def _prepare_grace_config(peft_config, model_config):
        if peft_config.grace_layer is None or peft_config.grace_config is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `grace_layer` and `grace_config` in `peft_config`")
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                else:
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
                    target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                                target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                                target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class VecDB:
    def __init__(self, grace_config):
        self.config = grace_config
        self.table = []

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        return self.table[item]

    def add_row(self, new_key, new_value, new_edit_label):
        new_row = {'key': None, 'value': None, 'radius': None, 'key_label': None}

        new_row['key'] = new_key.detach()
        new_row['value'] = new_value
        new_row['radius'] = torch.tensor(self.config['init_radius'], device=new_key.device).view(1)
        new_row['key_label'] = new_edit_label

        self.table.append(new_row)

    def update_row(self, index, key=None, value=None, radius=None, key_label=None):
        if key is not None:
            self.table[index]['key'] = key
        if value is not None:
            self.table[index]['value'] = value
        if radius is not None:
            self.table[index]['radius'] = radius
        if key_label is not None:
            self.table[index]['key_label'] = key_label

    def label_match(self, edit_label, key_label):
        edit_label = edit_label.masked_fill(edit_label == -100, 0)
        key_label = key_label.masked_fill(key_label == -100, 0)
        return torch.sum(edit_label) == torch.sum(key_label)

    def split_radii_in_half(self, nearest_key, smallest_distance):
        self.table[nearest_key]['radius'] = (smallest_distance / 2) - 1e-5
        self.table[-1]['radius'] = (smallest_distance / 2) + 1e-5

    def euc(self, batch_query, key):
        # Euclidean distance
        if len(key.shape) < 2:
            key = key.view(1, -1)
        return torch.cdist(batch_query, key, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    def search_database(self, batch_query):
        dists = []
        for i in range(len(self.table)):
            dists.append(self.euc(batch_query, self.table[i]['key']).view(-1, 1))
        return torch.stack(dists).view(-1, len(batch_query))


class GraceLayer:
    def __init__(self, grace_config: dict, in_features: int, out_features: int, **kwargs):
        self.grace_config = grace_config
        self.batch_iter = None
        for k, v in grace_config.items():
            setattr(self, k, v)
        self.VecDB = VecDB(grace_config)
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.lora_block_mapping = []
        self.non_overlap_edit = 0
        self.disable_grace = False
        self.block_id = 0

    def search(self, batch_query):
        dists = self.VecDB.search_database(batch_query)
        smallest_distance, nearest_key = dists.min(0)
        cached_view = [self.VecDB[i.item()] for i in nearest_key]
        lora_block_mapping = []
        for sd, row in zip(smallest_distance, cached_view):
            if sd > row['radius']:
                lora_block_mapping.append(NO_LORA)
            else:
                lora_block_mapping.append(row['value'])
        return smallest_distance, nearest_key, lora_block_mapping

    def current_block(self):
        return self.block_id

    def init_key_value(self, batch_query):
        lora_block_mapping = []
        for index, query in enumerate(batch_query):
            new_key = query.detach()
            new_eidt_label = self.edit_label[index]
            new_value = self.current_block()
            self.VecDB.add_row(new_key=new_key, new_value=new_value, new_edit_label=new_eidt_label)
            lora_block_mapping.append(new_value)
        self.block_id += 1
        return lora_block_mapping

    def add_key(self, query, label_index):
        new_key = query.detach()
        new_value = self.current_block()
        new_edit_label = self.edit_label[label_index]
        self.VecDB.add_row(new_key=new_key, new_value=new_value, new_edit_label=new_edit_label)
        return new_value

    def update_key(self, index, query=None, value=None, radius=None, key_label=None):
        key = None
        if query is not None:
            key = query.detach()
        self.VecDB.update_row(index, key, value, radius, key_label)


class GraceLinear(nn.Linear, GraceLayer):
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            grace_config: dict,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GraceLayer.__init__(self, grace_config=grace_config, in_features=in_features, out_features=out_features)
        self.fan_in_fan_out = fan_in_fan_out

    def forward(self, x: torch.Tensor):
        global LORA_BLOCK_MAPPING
        layer_out = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_grace:
            return layer_out

        '''Search Vector Database
        '''
        key_id = min(self.key_id, layer_out.shape[1] - 1)
        batch_query = layer_out[:, key_id]
        smallest_distance, nearest_key, lora_block_mapping = None, None, [NO_LORA] * layer_out.shape[0]
        if len(self.VecDB) != 0:
            smallest_distance, nearest_key, lora_block_mapping = self.search(batch_query)

        if not self.training:
            self.lora_block_mapping = LORA_BLOCK_MAPPING = lora_block_mapping
            return layer_out

        '''Only update the vector database once per batch
        '''
        if len(self.VecDB) == 0:
            lora_block_mapping = self.init_key_value(batch_query)
        elif self.batch_iter == 0:
            for index, query in enumerate(batch_query):
                row = self.VecDB[nearest_key[index]]
                if smallest_distance[index] > row['radius'] + self.init_radius:
                    # If there's no close key, make a new key, and reset the lora_block
                    lora_block_mapping[index] = self.add_key(query, label_index=index)
                elif self.VecDB.label_match(self.edit_label[index], row['key_label']):
                    print(f'The {index}th query is close to a previous edit, the labels are the same')
                    # If the current label is the SAME as the nearest label, just make the nearest epsilon bigger
                    # Avoid Catastrophic Forgetting
                    lora_block_mapping[index] = self.block_id
                    if self.expand_mode == "coverage":
                        self.update_key(nearest_key[index], query, self.block_id, smallest_distance[index])
                    elif self.expand_mode == "moving_avg":
                        a = 0.5
                        new_radius = a * row['radius'] + (1 - a) * smallest_distance[index]
                        new_key = a * row['key'] + (1 - a) * query
                        self.update_key(nearest_key[index], new_key, self.block_id, new_radius)
                else:
                    print(f'The {index}th query is close to a previous edit, but the labels are different')
                    lora_block_mapping[index] = self.add_key(query, label_index=index)
                    self.VecDB.split_radii_in_half(nearest_key[index], smallest_distance[index])
            self.block_id += 1
        else:
            pass

        self.lora_block_mapping = lora_block_mapping
        LORA_BLOCK_MAPPING = self.lora_block_mapping
        self.chosen_key = nearest_key
        # if self.training:
        #     assert NO_LORA not in LORA_BLOCK_MAPPING, "Training mode but no lora_block"
        return layer_out


class dynamic(nn.Module):
    def __init__(
            self,
            maximum_rank: int = 1,
            num_rank_per_block: int = 1
    ):
        assert maximum_rank % num_rank_per_block == 0, \
            "Maximum_rank % num_rank_per_block == 0 should be True"
        super(dynamic, self).__init__()
        self.maximum_rank = maximum_rank
        self.num_rank_per_block = num_rank_per_block
        self.maximum_block = maximum_rank // num_rank_per_block
        self.current_block = 0

    def get_block_dimension(self):
        return self.maximum_block

    def get_block(self):
        return self.current_block

    def set_block(self, block):
        self.current_block = max(0, min(block, self.get_block()))

    def block_rank_mapping(self, block_id):
        start = block_id * self.num_rank_per_block
        end = start + self.num_rank_per_block
        return start, end

    def update_dynamic(self, maximum_rank, num_rank_per_block):
        assert maximum_rank % num_rank_per_block == 0, \
            "Maximum_rank % num_rank_per_block == 0 should be True"
        self.maximum_rank = maximum_rank
        self.num_rank_per_block = num_rank_per_block
        self.maximum_block = maximum_rank // num_rank_per_block
        self.current_block = 0

    def forward(self, inputs):
        block_list = []
        assert len(LORA_BLOCK_MAPPING) != 0, "No element in LORA_BLOCK_MAPPING"
        for block_id in LORA_BLOCK_MAPPING:
            if block_id == NO_LORA:
                zero_tensor = torch.zeros((self.num_rank_per_block, inputs.shape[1]), device=inputs.device)
                block_list.append(zero_tensor)
            else:
                start, end = self.block_rank_mapping(block_id)
                block_list.append(inputs[start:end])
        result = torch.stack(block_list, 0)

        return result * math.sqrt(self.maximum_rank / self.num_rank_per_block)


class LoraLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self.nd_lora_A = dynamic()
        self.nd_lora_B = dynamic()

        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_rank_per_block):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.in_features, r)))})
            self.lora_B = nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.out_features)))})
            self.nd_lora_A.update_dynamic(r, num_rank_per_block)
            self.nd_lora_B.update_dynamic(r, num_rank_per_block)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            num_rank_per_block: int = 1,
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_rank_per_block)
        self.lora_block_mapping = LORA_BLOCK_MAPPING
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                    transpose(
                        self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                    transpose(
                        self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            lora_A = self.nd_lora_A(self.lora_A[self.active_adapter].T).mT
            lora_B = self.nd_lora_B(self.lora_B[self.active_adapter])
            result += (self.lora_dropout[self.active_adapter](x) @ lora_A @ lora_B) \
                      * self.scaling[self.active_adapter]

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class Embedding(nn.Embedding, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
            self,
            adapter_name: str,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                    transpose(
                        self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                    transpose(
                        self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                        transpose(
                            self.lora_embedding_B[self.active_adapter].weight
                            @ self.lora_embedding_A[self.active_adapter].weight,
                            True,
                        )
                        * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


class Conv2d(nn.Conv2d, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
            self,
            adapter_name: str,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]],
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int]] = 0,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        LoraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Conv2d.reset_parameters(self)
        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                self.weight.data += (
                                            self.lora_B[self.active_adapter].weight.squeeze(3).squeeze(2)
                                            @ self.lora_A[self.active_adapter].weight.squeeze(3).squeeze(2)
                                    ).unsqueeze(2).unsqueeze(3) * self.scaling[self.active_adapter]
            else:
                # conv2d 3x3
                self.weight.data += (
                        F.conv2d(
                            self.lora_A[self.active_adapter].weight.permute(1, 0, 2, 3),
                            self.lora_B[self.active_adapter].weight,
                        ).permute(1, 0, 2, 3)
                        * self.scaling[self.active_adapter]
                )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                self.weight.data -= (
                                            self.lora_B[self.active_adapter].weight.squeeze(3).squeeze(2)
                                            @ self.lora_A[self.active_adapter].weight.squeeze(3).squeeze(2)
                                    ).unsqueeze(2).unsqueeze(3) * self.scaling[self.active_adapter]
            else:
                # conv2d 3x3
                self.weight.data += (
                        F.conv2d(
                            self.lora_A[self.active_adapter].weight.permute(1, 0, 2, 3),
                            self.lora_B[self.active_adapter].weight,
                        ).permute(1, 0, 2, 3)
                        * self.scaling[self.active_adapter]
                )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            result += (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
            )
        else:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            ).to(expected_dtype)
                            * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                    )
                result += output
            return result


    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
            # Lora implemented in a dense layer
            def __init__(
                    self,
                    adapter_name,
                    in_features,
                    out_features,
                    r: int = 0,
                    lora_alpha: int = 1,
                    lora_dropout: float = 0.0,
                    **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_lora_weights = kwargs.pop("init_lora_weights", True)
                self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                        output = (
                                self.lora_B[self.active_adapter](
                                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                                ).to(expected_dtype)
                                * self.scaling[self.active_adapter]
                        )
                    else:
                        output = (
                                self.lora_B[self.active_adapter](
                                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                                )
                                * self.scaling[self.active_adapter]
                        )
                    result += output
                return result


