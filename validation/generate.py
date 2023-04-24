import argparse
import sys

import torch
import transformers
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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, help='the data used for instructing tuning')
parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'bloom'])
parser.add_argument('--size', type=str, help='the size of llama model')
parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)
args = parser.parse_args()

LOAD_8BIT = True
if args.model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    LORA_WEIGHTS = "./saved-"+args.data+args.size+"b"
elif args.model_type == "bloom":
    tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path)
    LORA_WEIGHTS = "/home/jovyan/gpt/open_gpt/alpaca-7b-chinese/finetuned/bloom-7b1-mt_medical-qa-instruction"
elif args.model_type == "chatglm":
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    LORA_WEIGHTS = "./saved_chatglm" + args.data 



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
        )
    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
        )
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
        )
elif device == "mps":
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
else:
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )

    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            args.model_name_or_path, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )   
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            args.model_name_or_path,trust_remote_code=True,
            device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )   
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=512,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()

"""
gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="alpaca4",
    description="Alpaca4",
).launch()

# Old testing code follows.

"""
if __name__ == "__main__":
    # testing code for readme
    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    while True:
        print("請輸入 Instruction:")
        instruction = input()
        print("請輸入 Input:")
        _input = input()
        response = evaluate(instruction, _input)
        if response[-4:] == "</s>":
            response = response[:-4]
        print("Response:", response)
        print()

