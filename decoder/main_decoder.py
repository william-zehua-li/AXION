# axion/decoder/main_decoder.py

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class MainDecoderOutput:
    internal_text: str
    hidden_state: Optional[torch.Tensor]
    memory_tokens: List[str]
    metadata: Dict[str, Any]


class MainDecoder:
    """
    Main decoder for AXION.

    It uses an existing causal language model and simulates memory nodes
    by prepending special memory tokens to the input.

    This decoder does not need to produce human-friendly text.
    It produces an internal reasoning/state representation for OutputDecoder.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        memory_token_count: int = 4,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.memory_token_count = memory_token_count
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.memory_tokens = [f"<MEM_{i}>" for i in range(memory_token_count)]
        self.special_tokens = {"additional_special_tokens": self.memory_tokens}
        self.tokenizer.add_special_tokens(self.special_tokens)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(self.device)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def build_prompt(
        self,
        user_input: str,
        prepro_signal: Dict[str, Any],
        focus_vector: Optional[str] = None,
        evidence_pack: Optional[str] = None,
        curr_state: Optional[str] = None,
        bias_profile: Optional[str] = None,
    ) -> str:
        memory_prefix = " ".join(self.memory_tokens)

        prompt = f"""
{memory_prefix}

[AXION_INTERNAL_MODE]
You are the main decoder.
Do not write a final human answer.
Produce compact internal state for the output decoder.

[PREPRO_SIGNAL]
{prepro_signal}

[FOCUS_VECTOR]
{focus_vector or "general_reasoning"}

[CURRENT_STATE]
{curr_state or "none"}

[BIAS_PROFILE]
{bias_profile or "none"}

[EVIDENCE_PACK]
{evidence_pack or "none"}

[USER_INPUT]
{user_input}

[OUTPUT_FORMAT]
Return internal state only:
- task_understanding:
- useful_evidence:
- uncertainty:
- reasoning_state:
- answer_intent:
[/AXION_INTERNAL_MODE]
""".strip()

        return prompt

    @torch.no_grad()
    def run(
        self,
        user_input: str,
        prepro_signal: Dict[str, Any],
        focus_vector: Optional[str] = None,
        evidence_pack: Optional[str] = None,
        curr_state: Optional[str] = None,
        bias_profile: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> MainDecoderOutput:
        prompt = self.build_prompt(
            user_input=user_input,
            prepro_signal=prepro_signal,
            focus_vector=focus_vector,
            evidence_pack=evidence_pack,
            curr_state=curr_state,
            bias_profile=bias_profile,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        generated_ids = outputs.sequences[0]
        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )

        # Remove prompt part if possible
        internal_text = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

        hidden_state = None
        if outputs.hidden_states:
            # Last generated step, last layer hidden state
            hidden_state = outputs.hidden_states[-1][-1]

        return MainDecoderOutput(
            internal_text=internal_text,
            hidden_state=hidden_state,
            memory_tokens=self.memory_tokens,
            metadata={
                "model_name": self.model_name,
                "focus_vector": focus_vector,
                "device": self.device,
            },
        )