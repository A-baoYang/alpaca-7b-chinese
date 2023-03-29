import json
import random

from opencc import OpenCC

converter = OpenCC("s2twp")  # 轉換為繁體及台灣常用詞彙

with open(
    "/home/jovyan/gpt/open_gpt/alpaca-7b-chinese/data/alpaca-zhCN.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

data = [
    {
        "instruction": converter.convert(item["instruction"]),
        "input": converter.convert(item["input"]),
        "output": converter.convert(item["output"]),
    }
    for item in data
]
random.shuffle(data)

with open(
    "/home/jovyan/gpt/open_gpt/alpaca-7b-chinese/data/alpaca-zhTW.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
