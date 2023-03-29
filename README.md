# Finetune LLaMA-7B with Chinese instruction dataset

## Installation

1. Install requirements

```bash
$ pip install -r requirements.txt
```

2. Install PyTorch at compatible version with CUDA

```bash
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Dataset

Combined all datasets using English-instruction, Chinese-output construction:

1. Translate [carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset) to Traditional Chinese with `OpenCC` package.

2. Addin Traditional Chinese dataset translate by ChatGPT API (`gpt-3.5-turbo`) by [ntunlplab/traditional-chinese-alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca) (Update at 2023.03.29)


## Finetune

1. Use finetune method provide by [tloen/alpaca-lora]

2. Use `torchrun` for distributed training

### Domain Tasks

(In progress)

1. 

## Serving

1. Provide FastAPI

2. Provide Dash 

