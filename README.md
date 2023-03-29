# 🦙🧋🇹🇼 Finetune LLaMA-7B with Chinese instruction datasets

This repository is a tutorial for finetuning LLaMA-7B with Chinese datasets! 
I survey and combine the dataset & method for finetuning my own LLM for complex NLP tasks such as summarization, question answering, text generation, custom data augmentation, etc. 

Since the original Stanford Alpaca-7B finetune need lots of GPU resources, I focus on surveying the method with low GPU consumption.

So here's how to reproduce:


## Installation

1. Install requirements

```bash
$ pip install -r requirements.txt
```

2. Install PyTorch at compatible version with CUDA

```bash
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```


## Datasets

This repository combined all datasets using English-instruction, Chinese-output construction:

1. `alpaca_data.json`: Original dataset from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
2. `alpaca_data_cleansed.json`: Cleansing by [gururise/AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)
3. `alpaca-zhCN.json`: Translate by [carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset)
4. `alpaca-zhTW.json`: Translate to Traditional Chinese using `OpenCC`
5. `alpaca-en-zh.json`: Combine the English instruction/input and Chinese output by [ntunlplab/traditional-chinese-alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca): (Traditional Chinese dataset translate by ChatGPT API (`gpt-3.5-turbo`) by [ntunlplab/traditional-chinese-alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca) (Update at 2023.03.29))


## Finetune

1. Use finetune method provide by [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)

2. Use `torchrun` for distributed training

```bash
$ torchrun --standalone --nnodes=1 --nproc_per_node=4 finetune.py
```

![](https://i.imgur.com/Czw3AAx.png)

### Finetune Domain Tasks

(In progress, welcome to discuss together: `jiunyi.yang.abao@gmail.com`. I'd like to try tasks from different domains such as investment, fraud, e-commerce, law, healthcare, ...)


## Model Serving
To serve your own model service through API & simple website UI!

1. Provide Model API

    ![](https://i.imgur.com/lkJnZ92.png)

2. Provide demo UI

    ![](https://i.imgur.com/SnihV9H.png)


## Learn More 

I curated lots of method that try to run large language models with fewer GPU resources:

- PEFT
- LoRA
- FlexGen
...

See full list: [chatgpt-alternatives](https://github.com/A-baoYang/chatgpt-alternatives)
