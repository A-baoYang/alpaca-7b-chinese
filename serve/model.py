import sys

import click
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from utils import generate_prompt


def decide_model(args, device_map):
    ModelClass = namedtuple("ModelClass", ('tokenizer', 'model'))
    _MODEL_CLASSES = {
        "llama": ModelClass(**{
            "tokenizer": LlamaTokenizer,
            "model": LlamaForCausalLM,
        }),
        "chatglm": ModelClass(**{
            "tokenizer": AutoTokenizer, #ChatGLMTokenizer,
            "model":  AutoModel, #ChatGLMForConditionalGeneration,
        }),
        "bloom": ModelClass(**{
            "tokenizer": BloomTokenizerFast,
            "model": BloomForCausalLM,
        }),
        "Auto": ModelClass(**{
            "tokenizer": AutoTokenizer,
            "model": AutoModel,
        })
    }
    model_type = "Auto" if args.model_type not in ["llama", "bloom", "chatglm"] else args.model_type
    
    if model_type == "chatglm":
        tokenizer = _MODEL_CLASSES[model_type].tokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True
        )
        # todo: ChatGLMForConditionalGeneration revision
        model = _MODEL_CLASSES[model_type].model.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            device_map=device_map
        )
    else:
        tokenizer = _MODEL_CLASSES[model_type].tokenizer.from_pretrained(args.base_model)
        model = _MODEL_CLASSES[model_type].model.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map
        )

    if model_type == "llama":
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"  # Allow batched inference

    if device_map == "auto":
        model = PeftModel.from_pretrained(
            model,
            finetuned_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            finetuned_weights,
            device_map=device_map
        )
    return tokenizer, model


class ModelServe:
    def __init__(
        self,
        load_8bit: bool = True,
        model_type: str = "llama",
        base_model: str = "decapoda-research/llama-7b-hf",
        finetuned_weights: str = "../finetuned/llama-7b-hf_alpaca-en-zh",
    ):
        args = locals()
        namedtupler = namedtuple("args", tuple(list(args.keys())))
        local_args = namedtupler(**args)
        
        if torch.cuda.is_available():
            self.device = "cuda"
            self.device_map = "auto"
            # self.max_memory = {i: "15GIB" for i in range(torch.cuda.device_count())}
            # self.max_memory.update({"cpu": "30GB"})
        else:
            self.device = "cpu"
            self.device_map = {"": self.device}

        self.tokenizer, self.model = decide_model(args=local_args, device_map=self.device_map)
        
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
