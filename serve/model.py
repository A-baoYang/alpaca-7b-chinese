import sys

import click
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils import generate_prompt


class ModelServe:
    def __init__(
        self,
        load_8bit: bool = True,
        base_model: str = "/home/jovyan/gpt/model/decapoda-research/llama-7b-hf",
        finetuned_weights: str = "/home/jovyan/gpt/open_gpt/alpaca-lora/finetuned/llama-7b-hf_alpaca-zh",
    ):
        if torch.cuda.is_available():
            self.device = "cuda"
            # self.max_memory = {i: "15GIB" for i in range(torch.cuda.device_count())}
            # self.max_memory.update({"cpu": "30GB"})
        else:
            self.device = "cpu"
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if self.device == "cuda":
            self.model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                finetuned_weights,
                torch_dtype=torch.float16,
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": self.device}, low_cpu_mem_usage=True
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                finetuned_weights,
                device_map={"": self.device},
            )

        # unwind broken decapoda-research config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if not load_8bit:
            self.model.half()  # seems to fix bugs for some users.

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def generate(
        self,
        instruction: str,
        input: str,
        temperature: float = 0.7,
        top_p: float = 0.75,
        top_k: int = 40,
        num_beams: int = 4,
        max_new_tokens: int = 1024,
        **kwargs
    ):
        prompt = generate_prompt(instruction, input)
        print(f"Prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        print("generating...")
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        print(f"Output: {output}")
        return output.split("### 回覆：")[1].strip()
