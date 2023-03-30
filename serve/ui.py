import gradio as gr
from model import ModelServe

model = ModelServe(load_8bit=False)
demo = gr.Interface(
    fn=model.generate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(
            minimum=0, maximum=100, step=1, value=40, label="Top k"
        ),
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
    title="🦙🌲 Alpaca-7B-Chinese",
    description="Alpaca-7B-Chinese is a 7B-parameter LLaMA model finetuned to follow instructions.",
)
demo.queue(concurrency_count=3)
demo.launch()
