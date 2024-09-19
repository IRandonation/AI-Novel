import random
import threading
import time

import gradio as gr
from Novel_Generation import AIGN  # 假设 chatLLM 在 novel_generator.py 中定义
from LLM import chatLLM

STREAM_INTERVAL = 0.

def display_memory_table(aign):
    """返回记忆的表格形式，用于展示在界面中"""
    character_status_df, relationships_df = aign.display_memory()
    return character_status_df, relationships_df

def expand_and_update_memory(aign, chapter_outline, paragraph_outline):
    """扩展段落并更新记忆，返回更新后的内容和记忆表格"""
    expanded_text = aign.expand_and_embellish_paragraph(chapter_outline, paragraph_outline)
    character_status_df, relationships_df = aign.display_memory()
    return expanded_text, character_status_df, relationships_df

with gr.Blocks() as demo:
    aign = gr.State(AIGN(chatLLM))
    gr.Markdown("## AI 写小说 Demo")

    with gr.Row():
        with gr.Column():
            chapter_outline_text = gr.Textbox(
                label="章节大纲",
                lines=4,
                interactive=True,
            )
            paragraph_outline_text = gr.Textbox(
                label="段落大纲",
                lines=4,
                interactive=True,
            )
            expand_button = gr.Button("扩展段落并润色")
            output_textbox = gr.Textbox(
                label="扩展内容",
                lines=10,
                interactive=False,
            )

        with gr.Column():
            character_status_table = gr.Dataframe(
                headers=["人物", "状态"],
                datatype=["str", "str"],
                label="人物状态",
            )
            relationships_table = gr.Dataframe(
                headers=["人物", "关系"],
                datatype=["str", "str"],
                label="人物关系",
            )

    # 绑定按钮事件
    expand_button.click(
        fn=expand_and_update_memory,
        inputs=[aign, chapter_outline_text, paragraph_outline_text],
        outputs=[output_textbox, character_status_table, relationships_table]
    )

demo.launch()
