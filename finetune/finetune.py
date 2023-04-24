# code 須改版成 ChatGLM-6b 那種

import os
import sys
from collections import namedtuple

import bitsandbytes as bnb
import click
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    LlamaTokenizer,
    set_seed,
)
from utils import PythonLiteralOption, check_distributed, generate_prompt, tokenize


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
            device_map=device_map
        )
    if model_type == "llama":
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return tokenizer, model


@click.command()
@click.option(
    "--base_model",
    "base_model",
    type=str,
    default="/home/jovyan/gpt/model/bigscience/bloomz-7b1-mt",
    # default="decapoda-research/llama-7b-hf",
)
@click.option("--model_type", "model_type", type=str, default="bloom")
@click.option(
    "--data_dir",
    "data_dir",
    type=str,
    default="/home/jovyan/gpt/open_gpt/alpaca-7b-chinese/data/medical/medical-qa-instruction-zhtw-test.json",
    # default="/home/jovyan/gpt/open_gpt/alpaca-7b-chinese/data/alpaca-en-zh.json",
)
@click.option(
    "--output_dir",
    "output_dir",
    type=str,
    default="/home/jovyan/gpt/open_gpt/alpaca-7b-chinese/finetuned/bloom-7b1-mt_medical-qa-instruction",
)
@click.option("--batch_size", "batch_size", type=int, default=128)
@click.option("--micro_batch_size", "micro_batch_size", type=int, default=1)
@click.option("--num_epochs", "num_epochs", type=int, default=20)
@click.option("--learning_rate", "learning_rate", type=float, default=3e-4)
@click.option("--cutoff_len", "cutoff_len", type=int, default=512)
@click.option("--val_set_size", "val_set_size", type=int, default=2000)
@click.option("--lora_r", "lora_r", type=int, default=8)
@click.option("--lora_alpha", "lora_alpha", type=int, default=16)
@click.option("--lora_dropout", "lora_dropout", type=float, default=0.05)
@click.option(
    "--lora_target_modules", "lora_target_modules", cls=PythonLiteralOption, default='["query_key_value"]', help="the module to be injected, e.g. q_proj/v_proj/k_proj/o_proj for llama, query_key_value for bloom"
)
@click.option("--train_on_inputs", "train_on_inputs", type=bool, default=True)
@click.option("--group_by_length", "group_by_length", type=bool, default=True)
def main(
    # model/data params
    base_model: str,
    model_type: str,
    data_dir: str,
    output_dir: str,
    # training hyperparams
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: int,
    # lora hyperparams
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list,
    # Trainer hyperparams
    train_on_inputs: bool,
    group_by_length: bool,
):  
    print(
        f"Finetune parameters: \n"
        f"base_model: {base_model}\n"
        f"model_type: {model_type}\n"
        f"data_dir: {data_dir}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
    )
    args = locals()
    namedtupler = namedtuple("args", tuple(list(args.keys())))
    local_args = namedtupler(**args)
    gradient_accumulation_steps = batch_size // micro_batch_size

    # setting torch distributed
    set_seed(888)
    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1
    print(rank, local_rank, world_size, is_distributed)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")
        device_map = {"": int(local_rank or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto"

    tokenizer, model = decide_model(args=local_args, device_map=device_map)
    data = load_dataset("json", data_files=data_dir)


    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(tokenizer, user_prompt, cutoff_len, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt


    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            dataloader_num_workers=world_size if world_size > 0 else 0,
            ddp_find_unused_parameters=False if is_distributed else None,
            group_by_length=group_by_length,
            # deepspeed="/home/jovyan/gpt/open_gpt/gptuner/config/zero_stage3_offload_config.json"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
