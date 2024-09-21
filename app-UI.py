import gradio as gr
import json
import os
from Novel_Generation import AIGN  # 假设 AIGN 在 Novel_Generation.py 中定义
from LLM import chatLLM

def generate_paragraph(aign, chapter_outline, paragraph_outline):
    """生成新段落并显示剧情总结"""
    paragraph = aign.generate_paragraph(chapter_outline, paragraph_outline)
    plot_summary = aign.get_memory_summary()  # 获取剧情总结
    return paragraph, plot_summary

def embellish_paragraph(aign, paragraph, embellishment_idea):
    """润色段落并更新剧情"""
    embellished_text = aign.embellish_paragraph(paragraph, embellishment_idea)
    plot_summary = aign.get_memory_summary()  # 获取剧情总结
    return embellished_text, plot_summary

def save_memory(aign):
    """保存当前记忆，提供一个可下载的文件"""
    memory_data = aign.get_memory_data()
    memory_json = json.dumps(memory_data, ensure_ascii=False, indent=4)
    
    # 创建临时文件保存路径
    file_path = "memory.json"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(memory_json)
    
    # 返回临时文件路径以供下载
    return file_path

def load_memory(aign, file_path):
    """从上传的文件加载记忆"""
    if file_path is not None:
        with open(file_path, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
        aign.load_memory_data(memory_data)  # 假设 AIGN 中有 load_memory_data 方法
        return "记忆已从上传的文件加载", aign.get_memory_summary()
    else:
        return "请上传有效的记忆文件", aign.get_memory_summary()

# 创建 Gradio 界面
with gr.Blocks() as demo:
    aign_state = gr.State(AIGN(chatLLM))
    gr.Markdown("## AI 小说写作助手")

    with gr.Row():
        with gr.Column():
            outline_text = gr.Textbox(label="章节或情节大纲", lines=4, interactive=True)
            expanded_outline_text = gr.Textbox(label="扩展大纲", lines=8, interactive=False)
            expand_button = gr.Button("扩展大纲")
            
            chapter_outline_text = gr.Textbox(label="章节大纲", lines=4, interactive=True)
            paragraph_outline_text = gr.Textbox(label="段落大纲", lines=4, interactive=True)
            embellishment_idea_text = gr.Textbox(label="润色要求", lines=4, interactive=True)
            paragraph_output = gr.Textbox(label="生成的段落", lines=10, interactive=False)
            embellish_output = gr.Textbox(label="润色后的段落", lines=10, interactive=False)
            generate_button = gr.Button("生成段落")
            embellish_button = gr.Button("润色段落")

            save_button = gr.Button("保存记忆")
            load_button = gr.Button("加载记忆")
            download_memory = gr.File(label="下载记忆文件", interactive=False)
            upload_memory = gr.File(label="上传记忆文件", type="filepath", interactive=True)

        with gr.Column():
            plot_summary_output = gr.Textbox(label="剧情走向", lines=10, interactive=False)

    # 绑定扩展大纲事件
    expand_button.click(
        fn=lambda aign, outline: aign.expand_outline(outline),
        inputs=[aign_state, outline_text],
        outputs=[expanded_outline_text]
    )

    # 绑定生成段落事件
    generate_button.click(
        fn=generate_paragraph, 
        inputs=[aign_state, chapter_outline_text, paragraph_outline_text], 
        outputs=[paragraph_output, plot_summary_output]
    )

    # 绑定润色段落事件
    embellish_button.click(
        fn=embellish_paragraph, 
        inputs=[aign_state, paragraph_output, embellishment_idea_text], 
        outputs=[embellish_output, plot_summary_output]
    )

    # 绑定保存记忆事件
    save_button.click(
        fn=save_memory, 
        inputs=[aign_state], 
        outputs=[download_memory]
    )

    # 绑定加载记忆事件
    load_button.click(
        fn=load_memory, 
        inputs=[aign_state, upload_memory], 
        outputs=[gr.Textbox(label="加载状态"), plot_summary_output]
    )

demo.launch()
