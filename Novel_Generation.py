import pandas as pd
import time
import json

from AIGN_Prompt import *

def Retryer(func, max_retries=10):
    def wrapper(*args, **kwargs):
        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print("-" * 30 + f"\n失败：\n{e}\n" + "-" * 30)
                time.sleep(2.333)
        raise ValueError("失败")

    return wrapper

class MarkdownAgent:
    """专门应对输入输出都是md格式的情况，例如小说生成"""

    def __init__(
        self,
        chatLLM,
        sys_prompt: str,
        name: str,
        temperature=0.8,
        top_p=0.8,
        use_memory=False,
        first_replay="明白了。",
        is_speak=True,
    ) -> None:

        self.chatLLM = chatLLM
        self.sys_prompt = sys_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.use_memory = use_memory
        self.is_speak = is_speak

        self.history = [{"role": "user", "content": self.sys_prompt}]

        if first_replay:
            self.history.append({"role": "assistant", "content": first_replay})
        else:
            resp = chatLLM(messages=self.history)
            self.history.append({"role": "assistant", "content": resp["content"]})

    def query(self, user_input: str) -> str:
        resp = self.chatLLM(
            messages=self.history + [{"role": "user", "content": user_input}],
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.use_memory:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": resp["content"]})

        return resp

    def getOutput(self, input_content: str, output_keys: list) -> dict:
        """解析类md格式中 # key 的内容"""
        resp = self.query(input_content)
        output = resp["content"]

        # 输出调试信息
        print(f"模型生成的输出: {output}")

        lines = output.split("\n")
        sections = {}
        current_section = ""

        for line in lines:
            # 以 # 开头认为是新部分
            if line.startswith("#"):
                current_section = line[1:].strip()
                sections[current_section] = []
            else:
                if current_section:
                    sections[current_section].append(line.strip())

        for key in sections.keys():
            sections[key] = "\n".join(sections[key]).strip()

        for k in output_keys:
            if k not in sections or not sections[k]:
                print(f"错误：未能解析 {k} 在输出:\n{output}\n\n")
                raise ValueError(f"fail to parse {k} in output:\n{output}\n\n")

        return sections

    def invoke(self, inputs: dict, output_keys: list) -> dict:
        input_content = ""
        for k, v in inputs.items():
            if isinstance(v, str) and len(v) > 0:
                input_content += f"# {k}\n{v}\n\n"

        result = Retryer(self.getOutput)(input_content, output_keys)

        return result

    def clear_memory(self):
        if self.use_memory:
            self.history = self.history[:2]

class AIGN:
    def __init__(self, chatLLM):
        self.chatLLM = chatLLM
        self.plot_summaries = []  # 用于存储多轮次的剧情总结
        self.global_plot_setting = "这是一个关于..."  # 全局剧情设定
        self.novel_writer = MarkdownAgent(
            chatLLM=self.chatLLM,
            sys_prompt="请根据以下章节大纲和段落大纲扩展成完整的段落。并在生成的正文前使用'#段落'进行标记。",
            name="NovelWriter",
            temperature=0.81,
        )
        self.novel_embellisher = MarkdownAgent(
            chatLLM=self.chatLLM,
            sys_prompt="请润色以下内容，使其更生动形象，并且在生成正文前使用'#润色'标记。",
            name="NovelEmbellisher",
            temperature=0.92,
        )
        self.memory_extractor = MarkdownAgent(
            chatLLM=self.chatLLM,
            sys_prompt="请从以下文本中提取剧情走向（简洁总结）并以'#剧情'为标题进行标记。",
            name="MemoryExtractor",
            temperature=0.7,
        )
        self.outline_expander = MarkdownAgent(
            chatLLM=self.chatLLM,
            sys_prompt="请根据以下简短的章节或情节大纲，生成一个这一段情节或者章节的扩展大纲作为这一段情节的参考，并在生成的内容前只使用'#扩展'进行标记。",
            name="OutlineExpander",
            temperature=0.85,
        )

    def expand_outline(self, outline):
        """扩展章节或情节大纲"""
        resp = self.outline_expander.invoke(
            inputs={"简短大纲": outline},
            output_keys=["扩展"]
        )
        expanded_outline = resp["扩展"]
        return expanded_outline

    def extract_memory(self, text):
        """利用大模型从文本中提取剧情总结"""
        resp = self.memory_extractor.invoke(
            inputs={"文本": text},
            output_keys=["剧情"]
        )
        plot_summary = resp["剧情"]
        return plot_summary

    def updateMemory(self, text):
        """利用模型总结内容并更新剧情"""
        new_summary = self.extract_memory(text)
        self.plot_summaries.append(new_summary)  # 将新的剧情加入到多轮次剧情中

    def generate_paragraph(self, chapter_outline, paragraph_outline):
        """生成段落并更新剧情"""
        current_memory_summary = self.get_memory_summary()
        resp = self.novel_writer.invoke(
            inputs={
                "章节大纲": chapter_outline,
                "段落大纲": paragraph_outline,
                "前文剧情": current_memory_summary
            },
            output_keys=["段落"]
        )
        next_paragraph = resp["段落"]
        self.updateMemory(next_paragraph)  # 更新剧情
        return next_paragraph
    
    def embellish_paragraph(self, paragraph, embellishment_idea):
        """润色给定的段落并更新剧情"""
        current_memory_summary = self.get_memory_summary()
        resp = self.novel_embellisher.invoke(
            inputs={
                "要润色的内容": paragraph,
                "润色要求": embellishment_idea,
                "前文剧情": current_memory_summary
            },
            output_keys=["润色"]
        )
        embellished_paragraph = resp["润色"]
        self.updateMemory(embellished_paragraph)  # 更新剧情
        return embellished_paragraph

    def get_memory_summary(self):
        """获取当前的剧情记忆摘要，包括全局设定和多轮次的剧情总结"""
        summary = self.global_plot_setting + "\n"
        summary += "\n".join(self.plot_summaries[-3:])  # 获取最近的3次剧情总结
        return summary
