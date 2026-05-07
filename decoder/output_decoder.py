# axion/decoder/output_decoder.py

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .main_decoder import MainDecoderOutput


@dataclass
class OutputDecoderResult:
    answer: str
    metadata: Dict[str, Any]


class OutputDecoder:
    """
    Output decoder for AXION.

    It converts MainDecoder internal state into human-readable language.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def build_prompt(
        self,
        user_input: str,
        main_output: MainDecoderOutput,
        style_instruction: Optional[str] = None,
    ) -> str:
        prompt = f"""
[AXION_OUTPUT_MODE]
You are the output decoder.
Convert the internal AXION state into a clear human-readable answer.

[USER_INPUT]
{user_input}

[INTERNAL_STATE_FROM_MAIN_DECODER]
{main_output.internal_text}

[STYLE_INSTRUCTION]
{style_instruction or "Answer clearly, directly, and avoid overclaiming."}

[RULES]
- Do not expose hidden implementation details.
- If uncertainty is present, state it clearly.
- If evidence is insufficient, say so.
- Do not invent facts.
- Produce the final user-facing answer.

[FINAL_ANSWER]
""".strip()

        return prompt

    @torch.no_grad()
    def run(
        self,
        user_input: str,
        main_output: MainDecoderOutput,
        style_instruction: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> OutputDecoderResult:
        prompt = self.build_prompt(
            user_input=user_input,
            main_output=main_output,
            style_instruction=style_instruction,
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
        )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        answer = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

        return OutputDecoderResult(
            answer=answer,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
            },
        )