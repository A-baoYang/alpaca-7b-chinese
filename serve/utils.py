def generate_prompt(instruction, input=None):
    if input:
        return ("以下是一個描述任務的指令，以及一個與任務資訊相關的輸入。請撰寫一個能適當完成此任務指令的回覆\n\n"
                f"### 指令：\n{instruction}\n\n### 輸入：\n{input}\n\n### 回覆：")
    else:
        return ("以下是一個描述任務的指令。請撰寫一個能適當完成此任務指令的回覆\n\n"
                f"### 指令：\n{instruction}\n\n### 回覆：\n")
