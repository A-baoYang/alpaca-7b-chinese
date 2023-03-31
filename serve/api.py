import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from model import ModelServe
from pydantic import BaseModel

app = FastAPI(
    title="Chinese Alpaca API",
    description="Run Alpaca with API",
    version="0.0.1",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = ModelServe(load_8bit=False)


@app.get("/")
async def read_root():
    return {"message": "Chinese Alpaca API is ready"}


@app.post("/completion/")
async def completion(
    instruction: str = Query(description="給模型的角色設定或指令", example="為用戶生成三種可能的投資標的建議", required=True), 
    input: str = Query(description="給模型的 context 或輸入", example="用戶是剛畢業的碩士生", default=None), 
):
    res = model.generate(instruction=instruction, input=input)
    return res


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=9889)
