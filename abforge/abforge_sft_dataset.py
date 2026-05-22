"""SFT dataset for ABForge Task1 / Task2.

The parquet stores already-assembled `prompt` (the user message body) and
`response` (`<Think>...</Think>` + `<Result>` or `<Proposed_Plan>` block)
as flat strings. We wrap the prompt with the Qwen3 chat template at load
time with ``enable_thinking=False`` so the assistant header does not open
its native ``<think>`` block (we use our own ``<Think>`` tag instead),
matching the RL launcher's ``+data.chat_template_kwargs.enable_thinking=False``.

Ablation: when ``mask_think_section=True`` is passed via the dataset config
(see ``run_abforge_qwen3_8b_task2_sft_ablation_no_think.sh``), the SFT loss
is computed only over the ``<Result>`` / ``<Proposed_Plan>`` portion: the
``<Think>...</Think>`` span (including the tags) is excluded from the loss
mask, so the model is not supervised on the audit-revised reasoning trace.
This realises the "w/o audit-in-the-loop" ablation by removing think
distillation entirely while keeping the SFT-only training recipe identical.
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class AbforgeSFTDataset(Dataset):
    """Single-turn SFT dataset that disables Qwen3 native thinking."""

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        response_key = config.get("response_key", "response")
        max_length = config.get("max_length", 6144)
        truncation = config.get("truncation", "right")
        chat_template_kwargs = config.get("chat_template_kwargs", None)
        if chat_template_kwargs is None:
            chat_template_kwargs = {"enable_thinking": False}
        else:
            # OmegaConf DictConfig -> plain dict
            chat_template_kwargs = dict(chat_template_kwargs)
        mask_think_section = bool(config.get("mask_think_section", False))
        think_open_tag = config.get("think_open_tag", "<Think>")
        think_close_tag = config.get("think_close_tag", "</Think>")

        assert truncation in ("error", "left", "right")
        self.truncation = truncation
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.chat_template_kwargs = chat_template_kwargs
        self.mask_think_section = mask_think_section
        self.think_open_tag = think_open_tag
        self.think_close_tag = think_close_tag
        self._think_mask_stats = {"checked": 0, "masked": 0, "no_tag": 0}

        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files()

    def _download(self):
        for i, path in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(path, verbose=True)

    def _read_files(self):
        frames = [pd.read_parquet(p) for p in self.parquet_files]
        self.df = pd.concat(frames).reset_index(drop=True)
        self.prompts = self.df[self.prompt_key].tolist()
        self.responses = self.df[self.response_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        prompt = self.prompts[item]
        response = self.responses[item]

        chat = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **self.chat_template_kwargs,
        )
        response_str = response + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if self.mask_think_section:
            resp_enc = tokenizer(
                response_str,
                return_tensors="pt",
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            response_ids = resp_enc["input_ids"][0]
            response_offsets = resp_enc["offset_mapping"][0]
        else:
            response_ids = tokenizer(response_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            response_offsets = None
        prompt_mask = torch.ones_like(prompt_ids)
        response_mask = torch.ones_like(response_ids)

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)

        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            pad_ids = torch.full(
                (pad_len,),
                fill_value=self.tokenizer.pad_token_id,
                dtype=input_ids.dtype,
            )
            pad_mask = torch.zeros(pad_len, dtype=attention_mask.dtype)
            input_ids = torch.cat((input_ids, pad_ids))
            attention_mask = torch.cat((attention_mask, pad_mask))
        elif seq_len > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_length {self.max_length}"
                )

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        if self.mask_think_section and response_offsets is not None:
            self._mask_think_tokens(loss_mask, response_str, response_offsets, prompt_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def _mask_think_tokens(self, loss_mask, response_str, response_offsets, prompt_length):
        """Zero out loss_mask positions corresponding to <Think>...</Think> tokens.

        We use char-level offset mapping (fast tokenizer) to locate every token
        whose [start, end) range falls inside the think span, then also mask
        the position immediately before the first think token so the model is
        not supervised to *predict* the opening of the think block either.
        """
        i_open = response_str.find(self.think_open_tag)
        if i_open < 0:
            self._think_mask_stats["no_tag"] += 1
            return
        i_close_start = response_str.find(self.think_close_tag, i_open + len(self.think_open_tag))
        if i_close_start < 0:
            self._think_mask_stats["no_tag"] += 1
            return
        i_close_end = i_close_start + len(self.think_close_tag)

        starts = response_offsets[:, 0]
        ends = response_offsets[:, 1]
        in_think = (starts >= i_open) & (ends <= i_close_end) & (ends > starts)

        idxs = in_think.nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            self._think_mask_stats["no_tag"] += 1
            return

        cap = loss_mask.size(0)
        first_local = int(idxs[0].item())
        last_local = int(idxs[-1].item())

        first_global = prompt_length + first_local
        last_global = prompt_length + last_local
        prev_global = first_global - 1

        if 0 <= prev_global < cap:
            loss_mask[prev_global] = 0
        lo = max(0, first_global)
        hi = min(cap, last_global + 1)
        if lo < hi:
            loss_mask[lo:hi] = 0
        self._think_mask_stats["checked"] += 1
        self._think_mask_stats["masked"] += int(idxs.numel())
