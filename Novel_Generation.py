import pandas as pd
import time

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
        # first_replay=None,
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
            # if self.is_speak:
            #     self.speak(Msg(self.name, resp["content"]))

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
        """解析类md格式中 # key 的内容，未解析全部output_keys中的key会报错"""
        resp = self.query(input_content)
        output = resp["content"]

        lines = output.split("\n")
        sections = {}
        current_section = ""
        for line in lines:
            if line.startswith("# ") or line.startswith(" # "):
                # new key
                current_section = line[2:].strip()
                sections[current_section] = []
            else:
                # add content to current key
                if current_section:
                    sections[current_section].append(line.strip())
        for key in sections.keys():
            sections[key] = "\n".join(sections[key]).strip()

        for k in output_keys:
            if (k not in sections) or (len(sections[k]) == 0):
                raise ValueError(f"fail to parse {k} in output:\n{output}\n\n")

        # if self.is_speak:
        #     self.speak(
        #         Msg(
        #             self.name,
        #             f"total_tokens: {resp['total_tokens']}\n{resp['content']}\n",
        #         )
        #     )
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

class MemorySystem:
    def __init__(self):
        # 存储人物的状态和关系
        self.character_memory = {}
        self.relationships = {}

    def update_character(self, character_name, status):
        """更新人物的状态"""
        self.character_memory[character_name] = status

    def update_relationship(self, character1, character2, relationship):
        """更新人物之间的关系"""
        self.relationships[(character1, character2)] = relationship

    def get_character_status(self, character_name):
        """获取人物的当前状态"""
        return self.character_memory.get(character_name, "未知状态")

    def get_relationship(self, character1, character2):
        """获取人物之间的关系"""
        return self.relationships.get((character1, character2), "未知关系")

    def memory_summary(self):
        """返回当前所有人物和关系的记忆摘要"""
        summary = "当前人物状态：\n"
        for character, status in self.character_memory.items():
            summary += f"{character}: {status}\n"

        summary += "\n人物关系：\n"
        for (char1, char2), relation in self.relationships.items():
            summary += f"{char1} 和 {char2} 的关系: {relation}\n"

        return summary

    def update_memory_from_text(self, extracted_info):
        """根据大模型提取的信息更新人物状态和关系"""
        if "characters" in extracted_info:
            for character, status in extracted_info["characters"].items():
                self.update_character(character, status)

        if "relationships" in extracted_info:
            for (char1, char2), relation in extracted_info["relationships"].items():
                self.update_relationship(char1, char2, relation)

    def display_memory_as_table(self):
        """将记忆以表格形式显示"""
        # 创建人物状态表
        character_status_df = pd.DataFrame.from_dict(self.character_memory, orient='index', columns=['状态'])
        # 创建人物关系表
        relationships_df = pd.DataFrame(list(self.relationships.items()), columns=['人物', '关系'])
        return character_status_df, relationships_df


class AIGN:
    def __init__(self, chatLLM):
        self.chatLLM = chatLLM
        self.memory_system = MemorySystem()
        self.paragraph_list = []
        self.novel_content = ""
        self.embellishment_idea = ""

        self.novel_writer = MarkdownAgent(
            chatLLM=self.chatLLM,
            sys_prompt="请根据以下章节大纲和段落大纲扩展成完整的段落。",
            name="NovelWriter",
            temperature=0.81,
        )
        self.novel_embellisher = MarkdownAgent(
            chatLLM=self.chatLLM,
            sys_prompt="请润色以下内容，使其更生动形象。",
            name="NovelEmbellisher",
            temperature=0.92,
        )
        self.memory_extractor = MarkdownAgent(
            chatLLM=self.chatLLM,
            sys_prompt="请从以下文本中提取人物的状态和关系，并以JSON格式返回。例如：{'characters': {'角色A': '状态描述'}, 'relationships': {('角色A', '角色B'): '关系描述'}}。",
            name="MemoryExtractor",
            temperature=0.7,
        )

    def expand_and_embellish_paragraph(self, chapter_outline, paragraph_outline):
        """根据章节和段落大纲扩展和润色文本"""
        # 调用小说写作代理生成扩展段落
        resp = self.novel_writer.invoke(
            inputs={
                "章节大纲": chapter_outline,
                "段落大纲": paragraph_outline,
                "前文记忆": self.memory_system.memory_summary()
            },
            output_keys=["段落"]
        )
        next_paragraph = resp["段落"]

        # 调用润色代理进行润色
        resp = self.novel_embellisher.invoke(
            inputs={
                "要润色的内容": next_paragraph,
                "润色要求": self.embellishment_idea
            },
            output_keys=["润色结果"]
        )
        embellished_paragraph = resp["润色结果"]

        # 更新小说内容
        self.paragraph_list.append(embellished_paragraph)
        self.updateNovelContent()

        # 使用大模型提取人物状态和关系
        self.extract_memory_with_llm(embellished_paragraph)

        return embellished_paragraph

    def extract_memory_with_llm(self, text):
        """利用大模型从文本中提取人物状态和关系"""
        resp = self.memory_extractor.invoke(
            inputs={
                "文本": text
            },
            output_keys=["人物状态和关系"]
        )
        extracted_info = eval(resp["人物状态和关系"])  # 假设大模型返回的结果是字典格式
        self.memory_system.update_memory_from_text(extracted_info)

    def updateNovelContent(self):
        self.novel_content = ""
        for paragraph in self.paragraph_list:
            self.novel_content += f"{paragraph}\n\n"
        return self.novel_content

    def display_memory(self):
        """显示人物状态和关系表格"""
        character_status_df, relationships_df = self.memory_system.display_memory_as_table()
        return character_status_df, relationships_df
