import os
import sys
from typing import List

import bitsandbytes as bnb
import click
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from .utils import generate_and_tokenize_prompt, generate_prompt, tokenize


@click.command()
@click.option(
    "--base_model",
    "base_model",
    type=str,
    default="/home/jovyan/gpt/model/decapoda-research/llama-7b-hf",
    required=True,
)
@click.option(
    "--data_path",
    "data_path",
    type=str,
    default="./data/alpaca-en-zh.json",
    required=True,
)
@click.option(
    "--output_dir",
    "output_dir",
    type=str,
    default="./finetuned/llama-7b-hf_alpaca-en-zh",
    required=True,
)
@click.option("--batch_size", "batch_size", type=int, default=128)
@click.option("--micro_batch_size", "micro_batch_size", type=int, default=4)
@click.option("--num_epochs", "num_epochs", type=int, default=20)
@click.option("--learning_rate", "learning_rate", type=float, default=3e-4)
@click.option("--cutoff_len", "cutoff_len", type=int, default=512)
@click.option("--val_set_size", "val_set_size", type=int, default=2000)
@click.option("--lora_r", "lora_r", type=int, default=8)
@click.option("--lora_alpha", "lora_alpha", type=int, default=16)
@click.option("--lora_dropout", "lora_dropout", type=float, default=0.05)
@click.option(
    "--lora_target_modules", "lora_target_modules", type=List[str], default=20
)
@click.option("--train_on_inputs", "train_on_inputs", type=bool, default=True)
@click.option("--group_by_length", "group_by_length", type=bool, default=True)
def main(
    # model/data params
    base_model: str,
    data_path: str,
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
    lora_target_modules: List[str],
    # llm hyperparams
    train_on_inputs: bool,
    group_by_length: bool,
):
    print(
        f"Finetune parameters: \n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
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
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    data = load_dataset("json", data_files=data_path)

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
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
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
