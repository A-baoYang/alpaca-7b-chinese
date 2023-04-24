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

Reference finetune method provide by [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) 

1. Run on 1 GPU with Colab: https://colab.research.google.com/drive/1QvtrJpikkkNKSbwwG766SIGbBw2TQRd5?usp=sharing [![Open in Colab][Colab Badge]][Code Notebook]

  - `LLaMA`
    ```bash
    $ cd finetune/
    $ python finetune.py --base_model decapoda-research/llama-7b-hf --data_dir ../data/alpaca-en-zh.json --output_dir ../finetuned/llama-7b-hf_alpaca-en-zh --lora_target_modules '["q_proj", "v_proj"]'
    ```
  
  - `BLOOM`
    ```bash
    $ cd finetune/
    $ python finetune.py --base_model bigscience/bloomz-7b1-mt --data_dir ../data/alpaca-en-zh.json --output_dir ../finetuned/bloomz-7b1-mt_alpaca-en-zh --lora_target_modules '["query_key_value"]'
    ```

2. Use `torchrun` for distributed training on Multi-GPUs

  - `LLaMA`
    ```bash
    $ cd finetune/
    $ torchrun --standalone --nnodes=1 --nproc_per_node=4 finetune.py --base_model decapoda-research/llama-7b-hf --data_dir ../data/alpaca-en-zh.json --output_dir ../finetuned/llama-7b-hf_alpaca-en-zh --lora_target_modules '["q_proj", "v_proj"]'
    ```

  - `BLOOM`
    ```bash
    $ cd finetune/
    $ torchrun --standalone --nnodes=1 --nproc_per_node=4 finetune.py --base_model bigscience/bloomz-7b1-mt --data_dir ../data/alpaca-en-zh.json --output_dir ../finetuned/bloomz-7b1-mt_alpaca-en-zh --lora_target_modules '["query_key_value"]'
    ```

![](https://i.imgur.com/Czw3AAx.png)

### Finetune Domain Tasks

I've collected different domain tasks in my repository: [instruction-finetune-datasets](https://github.com/A-baoYang/instruction-finetune-datasets)

Welcome cooperations! Please contact me at: `jiunyi.yang.abao@gmail.com`. I'd like to try tasks from different domains such as investment, fraud, e-commerce, law, healthcare, ...


## Model Serving
To serve your own model service through API & simple website UI!

1. Model API

    ![](https://i.imgur.com/lkJnZ92.png)

    ```bash
    $ cd serve/
    $ python api.py
    ```

2. demo UI

    ![](https://i.imgur.com/SnihV9H.png)

    ```bash
    $ cd serve/
    $ python ui.py
    ```


## Learn More 

I curated lots of method that try to run large language models with fewer GPU resources:

- PEFT
- LoRA
- FlexGen
...

See full list: [chatgpt-alternatives](https://github.com/A-baoYang/chatgpt-alternatives)


```
@misc{alpaca-7b-chinese,
  author = {JiunYi Yang},
  title = {Alpaca-7B Chinese: Finetune LLaMA-7B with Chinese instruction datasets},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/A-baoYang/alpaca-7b-chinese}},
}
```
