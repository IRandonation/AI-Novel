import random
import threading
import time

import gradio as gr
from Novel_Generation import AIGN  # 假设 chatLLM 在 novel_generator.py 中定义
from LLM import chatLLM

STREAM_INTERVAL = 0.

def generate_paragraph(aign, chapter_outline, paragraph_outline):
    """生成新段落并显示"""
    paragraph = aign.generate_paragraph(chapter_outline, paragraph_outline)
    return paragraph

def embellish_paragraph(aign, paragraph):
    """润色段落并更新记忆"""
    embellished_text = aign.embellish_paragraph(paragraph)
    # aign.extract_memory_with_llm(embellished_text)  # 更新记忆
    # character_status_df, relationships_df = aign.display_memory()
    # 带人物状态 return embellished_text, character_status_df, relationships_df
    return embellished_text

# 创建 Gradio 界面
with gr.Blocks() as demo:
    aign_state = gr.State(AIGN(chatLLM))
    gr.Markdown("## AI 小说写作助手")

    with gr.Row():
        with gr.Column():
            chapter_outline_text = gr.Textbox(label="章节大纲", lines=4, interactive=True)
            paragraph_outline_text = gr.Textbox(label="段落大纲", lines=4, interactive=True)
            paragraph_output = gr.Textbox(label="生成的段落", lines=10, interactive=False)
            embellish_output = gr.Textbox(label="润色后的段落", lines=10, interactive=False)
            generate_button = gr.Button("生成段落")
            embellish_button = gr.Button("润色段落")

        # with gr.Column():
        #     character_status_table = gr.Dataframe(headers=["人物", "状态"], datatype=["str", "str"], label="人物状态")
        #     relationships_table = gr.Dataframe(headers=["人物", "关系"], datatype=["str", "str"], label="人物关系")

    # 绑定生成段落事件
    generate_button.click(fn=generate_paragraph, inputs=[aign_state, chapter_outline_text, paragraph_outline_text], outputs=[paragraph_output])

    # 绑定润色段落事件
    # 加上人物关系：embellish_button.click(fn=embellish_paragraph, inputs=[aign_state, paragraph_output], outputs=[embellish_output, character_status_table, relationships_table])
    embellish_button.click(fn=embellish_paragraph, inputs=[aign_state, paragraph_output], outputs=[embellish_output])

demo.launch()
