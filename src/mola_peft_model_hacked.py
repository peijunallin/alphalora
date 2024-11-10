# coding=utf-8
from __future__ import annotations

import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin

from peft import __version__
from peft.tuners import (
    AdaLoraModel,
    AdaptionPromptModel,
    # LoraModel,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)
##newmodified
from src.mola_lora_hacked import LoraModel

from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    TaskType,
    _set_adapter,
    _set_trainable,
    add_library_to_model_card,
    get_peft_model_state_dict,
    hub_file_exists,
    set_peft_model_state_dict,
    shift_tokens_right,
)


PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.PROMPT_TUNING: PromptEmbedding,
    PeftType.P_TUNING: PromptEncoder,
    PeftType.PREFIX_TUNING: PrefixEncoder,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.ADAPTION_PROMPT: AdaptionPromptModel,
}

##newmodified3

def get_peft_model_state_dict_moe(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            # to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
            to_return = {}
            for k in state_dict:
                if "lora_" in k or "router" in k:
                    to_return[k] = state_dict[k]
                    if "router" in k:
                        print("save*****", k)
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("router" in k) or ("bias" in k))}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif isinstance(config, PromptLearningConfig):
        to_return = {}
        if config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
    elif config.peft_type == PeftType.IA3:
        to_return = {k: state_dict[k] for k in state_dict if "ia3_" in k}
    else:
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return
##newmodified4
def set_peft_model_state_dict_moe(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.IA3):
        peft_model_state_dict = {}
        parameter_prefix = "ia3_" if config.peft_type == PeftType.IA3 else "lora_"
        for k, v in state_dict.items():
            if parameter_prefix in k:
                suffix = k.split(parameter_prefix)[1]
                experts_id = suffix.split(".")[0].split("_")[1]
                if "." in suffix:
                    # suffix_to_replace = ".".join(suffix.split(".")[1:])
                    # k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix[1:], f".{adapter_name}_{experts_id}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                if "router" in k:
                    print("load-----",k)
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif isinstance(config, PromptLearningConfig) or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if isinstance(config, PromptLearningConfig):
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
    return load_result

class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.


    **Attributes**:
        - **base_model** ([`~transformers.PreTrainedModel`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
        saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
        using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
        using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
        backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
        in the base model if using [`PromptLearningConfig`].
    """

    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default",
                 number_experts: list = [8] * 32,
                 top_k: list = [2] * 32):
        super().__init__()
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.base_model_torch_dtype = getattr(model, "dtype", None)
        self.number_experts = number_experts
        self.top_k = top_k
        if not isinstance(peft_config, PromptLearningConfig):
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name, self.number_experts, self.top_k
            )
            self.set_additional_trainable_modules(peft_config, adapter_name)
        else:
            self.add_adapter(adapter_name, peft_config)

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

    def save_pretrained(self, save_directory: str, safe_serialization: bool = False, **kwargs: Any):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        self.create_or_update_model_card(save_directory)
        ##newmodified3
        for adapter_name, peft_config in self.peft_config.items():
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict_moe(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if safe_serialization:
                safe_save_file(
                    output_state_dict, os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME), metadata={"format": "pt"}
                )
            else:
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if isinstance(peft_config, PromptLearningConfig)
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        number_experts: list = [8] * 8 + [6] * 8 + [4] * 8 + 2 * [8],
        top_k: list = [2] * 32,
        **kwargs: Any,
    ):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and use for
                inference
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuation. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific Lora configuration class.
        """
        from src.mola_mapping_hacked import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                )
            ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None), **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if isinstance(config, PromptLearningConfig) and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name)
        else:

            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name=adapter_name, number_experts=number_experts, top_k=top_k)
            # model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name, number_experts, top_k)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
        return model

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        if not (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def get_prompt_embedding_to_save(self, adapter_name: str):
        """
        Returns the prompt embedding to save when saving the model. Only applicable when `peft_config.peft_type !=
        PeftType.LORA`.
        """
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name].unsqueeze(0).expand(1, -1).to(prompt_encoder.embedding.weight.device)
        )
        if self.peft_config[adapter_name].peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config[adapter_name].num_virtual_tokens]
        prompt_embeddings = prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size: int):
        """
        Returns the virtual prompts to use for Peft. Only applicable when `peft_config.peft_type != PeftType.LORA`.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                peft_config.num_transformer_submodules * 2
            )
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if peft_config.inference_mode:
                prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = prompt_encoder(prompt_tokens)
            return prompts

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """
        try:
            if isinstance(self.peft_config[self.active_adapter], PromptLearningConfig):
                old_forward = self.forward
                self.forward = self.base_model.forward
            else:
                self.base_model.disable_adapter_layers()
            yield
        finally:
            if isinstance(self.peft_config[self.active_adapter], PromptLearningConfig):
                self.forward = old_forward
            else:
                self.base_model.enable_adapter_layers()

    def get_base_model(self):
        """
        Returns the base model.
        """
        return self.base_model if isinstance(self.active_peft_config, PromptLearningConfig) else self.base_model.model

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig):
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )
        self.peft_config[adapter_name] = peft_config
        if isinstance(peft_config, PromptLearningConfig):
            self._setup_prompt_encoder(adapter_name)
        else:
            self.base_model.add_adapter(adapter_name, peft_config)

        self.set_additional_trainable_modules(peft_config, adapter_name)

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)

    @classmethod
    def _split_kwargs(cls, kwargs: Dict[str, Any]):
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def load_adapter(self, model_id: str, adapter_name: str, is_trainable: bool = False, **kwargs: Any):
        from src.mola_mapping_hacked import PEFT_TYPE_TO_CONFIG_MAPPING

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                )
            ].from_pretrained(
                model_id,
                subfolder=kwargs.get("subfolder", None),
                revision=kwargs.get("revision", None),
                cache_dir=kwargs.get("cache_dir", None),
            )
            if isinstance(peft_config, PromptLearningConfig) and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        # load weights if any
        path = os.path.join(model_id, kwargs["subfolder"]) if kwargs.get("subfolder", None) is not None else model_id

        if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
            filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
            use_safetensors = True
        elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
            use_safetensors = False
        else:
            has_remote_safetensors_file = hub_file_exists(
                model_id, SAFETENSORS_WEIGHTS_NAME, revision=kwargs.get("revision", None)
            )
            use_safetensors = has_remote_safetensors_file

            if has_remote_safetensors_file:
                # Priority 1: load safetensors weights
                filename = hf_hub_download(
                    model_id,
                    SAFETENSORS_WEIGHTS_NAME,
                    subfolder=kwargs.get("subfolder", None),
                    **hf_hub_download_kwargs,
                )
            else:
                try:
                    filename = hf_hub_download(
                        model_id, WEIGHTS_NAME, subfolder=kwargs.get("subfolder", None), **hf_hub_download_kwargs
                    )
                except EntryNotFoundError:
                    raise ValueError(
                        f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                        f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                    )

        if use_safetensors:
            adapters_weights = safe_load_file(filename, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            adapters_weights = torch.load(
                filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        # load the weights into the model
        ##newmodified3
        load_result = set_peft_model_state_dict_moe(self, adapters_weights, adapter_name=adapter_name)
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            if isinstance(self.peft_config[adapter_name], PromptLearningConfig):
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    def set_adapter(self, adapter_name: str):
        """
        Sets the active adapter.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        if not isinstance(self.peft_config[adapter_name], PromptLearningConfig):
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create model card to include information about peft:
        1. Adds `peft` library tag
        2. Adds peft version
        3. Adds quantization information if it was used
        """
        # Adds `peft` library tag
        add_library_to_model_card(output_dir)

        with open(os.path.join(output_dir, "README.md"), "r") as f:
            lines = f.readlines()

        quantization_config = None
        if hasattr(self.config, "quantization_config"):
            quantization_config = self.config.quantization_config.to_dict()
        training_config_text = ""
        # Adds quantization information if it was used
        if quantization_config is not None:
            training_config_text += "\nThe following `bitsandbytes` quantization config was used during training:\n"
            training_config_text += "\n".join([f"- {name}: {value}" for name, value in quantization_config.items()])
            training_config_text += "\n"

        training_procedure_heading = "## Training procedure\n"
        if training_procedure_heading in lines:
            lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
        else:
            lines.append(f"{training_procedure_heading}\n{training_config_text}")

        # Adds peft version
        framework_block_heading = "### Framework versions\n"
        if framework_block_heading in lines:
            lines.insert(lines.index(framework_block_heading) + 2, f"- PEFT {__version__}\n")
        else:
            lines.append(f"{framework_block_heading}\n\n- PEFT {__version__}\n")

        # write the lines back to README.md
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.writelines(lines)


class PeftModelForSequenceClassification(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import PeftModelForSequenceClassification, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "SEQ_CLS",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 768,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 12,
        ...     "num_layers": 12,
        ...     "encoder_hidden_size": 768,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForSequenceClassification(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.


    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default", number_experts= [8] * 32, top_k=[2] * 32):
        super().__init__(model, peft_config, adapter_name, number_experts= number_experts, top_k=top_k)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
    ##newmodified
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=True,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
            
            ##newmodified
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if isinstance(peft_config, PromptLearningConfig):
            if model_kwargs.get("attention_mask", None) is not None:
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs


class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for sequence-to-sequence language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.


    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import PeftModelForSeq2SeqLM, get_peft_config

        >>> config = {
        ...     "peft_type": "LORA",
        ...     "task_type": "SEQ_2_SEQ_LM",
        ...     "inference_mode": False,
        ...     "r": 8,
        ...     "target_modules": ["q", "v"],
        ...     "lora_alpha": 32,
        ...     "lora_dropout": 0.1,
        ...     "fan_in_fan_out": False,
        ...     "enable_lora": None,
        ...     "bias": "none",
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> peft_model = PeftModelForSeq2SeqLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                decoder_attention_mask.device
            )
            decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, **kwargs
            )
        elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)

            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # concat prompt labels
            if labels is not None:
                if peft_config.num_transformer_submodules == 1:
                    kwargs["labels"] = labels
                elif peft_config.num_transformer_submodules == 2:
                    prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                    kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
            if peft_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif peft_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = torch.cat(
                    (prompts[:, peft_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
                )
                return self.base_model(
                    inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs
                )

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if not isinstance(peft_config, PromptLearningConfig):
                outputs = self.base_model.generate(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                if peft_config.peft_type == PeftType.PREFIX_TUNING:
                    outputs = self.base_model.generate(**kwargs)
                elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
                    kwargs = deepcopy(kwargs)

                    if "encoder_outputs" in kwargs:
                        del kwargs["encoder_ouputs"]
                        warnings.warn(
                            "`encoder_outputs` should not be passed to `generate` when using prompt tuning. Ignoring it."
                        )

                    input_ids = kwargs.pop("input_ids")
                    inputs_embeds = self.word_embeddings(input_ids)
                    batch_size = inputs_embeds.shape[0]
                    prompts = self.get_prompt(batch_size=batch_size)
                    prompts = prompts.to(inputs_embeds.dtype)

                    inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
                    kwargs["inputs_embeds"] = inputs_embeds

                    if "attention_mask" in kwargs:
                        prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                            kwargs["attention_mask"].device
                        )
                        kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

                    return self.base_model.generate(**kwargs)
                else:
                    raise NotImplementedError
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            model_kwargs["past_key_values"] = past_key_values

        return model_kwargs


class PeftModelForTokenClassification(PeftModel):
    """
    Peft model for token classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import PeftModelForTokenClassification, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "TOKEN_CLS",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 768,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 12,
        ...     "num_layers": 12,
        ...     "encoder_hidden_size": 768,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForTokenClassification(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(self, model, peft_config: PeftConfig = None, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PeftModelForQuestionAnswering(PeftModel):
    """
    Peft model for extractive question answering.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForQuestionAnswering
        >>> from peft import PeftModelForQuestionAnswering, get_peft_config

        >>> config = {
        ...     "peft_type": "LORA",
        ...     "task_type": "QUESTION_ANS",
        ...     "inference_mode": False,
        ...     "r": 16,
        ...     "target_modules": ["query", "value"],
        ...     "lora_alpha": 32,
        ...     "lora_dropout": 0.05,
        ...     "fan_in_fan_out": False,
        ...     "bias": "none",
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForQuestionAnswering(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 592900 || all params: 108312580 || trainable%: 0.5473971721475013
        ```
    """

    def __init__(self, model, peft_config: PeftConfig = None, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"qa_outputs"}
        else:
            self.modules_to_save.update({"qa_outputs"})

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                start_positions=start_positions,
                end_positions=end_positions,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(start_positions=start_positions, end_positions=end_positions, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            total_loss = None
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
