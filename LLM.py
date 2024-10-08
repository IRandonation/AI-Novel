from uniai import aliChatLLM, deepseekChatLLM, zhipuChatLLM

chatLLM = zhipuChatLLM(model_name="GLM-4", api_key="93c8238ebf3fe2cf702cbc1ce5ec4d1b.xqkKBB72dleCYIjs")

if __name__ == "__main__":

    content = "请用一个成语介绍你自己"
    messages = [{"role": "user", "content": content}]

    resp = chatLLM(messages)
    print(resp)

    for resp in chatLLM(messages, stream=True):
        print(resp)
