from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from model_pool.promptTemplate import PROMPT_TEMPLATE_EN, PROMPT_TEMPLATE_ZH


class BaseModel:

    def chat(self, prompt, history, content):
        pass

    def load_model(self):
        pass

class ChatGLM(BaseModel):
    def __init__(self, model_path, max_memory_map = {0: "11GB", 1: "11GB"}):
        super().__init__()
        self.model_path = model_path
        self.max_memory_map = max_memory_map
        self.load_model()

    def load_model(self):
        print("Start loading ChatGLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, max_memory=self.max_memory_map, device_map='auto')
        self.model.eval()
        print("Finish model building!")

    def chat(self, question, history, prompt_template, context=None):
        if prompt_template == "CHATGLM_TEMPLATE" and context == None:
            prompt = PROMPT_TEMPLATE_ZH[prompt_template].format(question=question)
            print("CHATGLM_TEMPLATE: ", prompt)
        elif prompt_template == "RAG_CHATGLM_TEMPLATE" and context is not None:
            prompt = PROMPT_TEMPLATE_ZH[prompt_template].format(question=question, text=context)
            print("RAG_CHATGLM_TEMPLATE: ", prompt)
        elif prompt_template == "SUMMARY_CHATGLM_TEMPLATE" and context == None:
            prompt = PROMPT_TEMPLATE_ZH[prompt_template].format(text=question)
            print("SUMMARY_CHATGLM_TEMPLATE: ", prompt)
        response, history = self.model.chat(self.tokenizer, question, prompt, history)
        return response, history

    def query_transform(self, question, history):
        conversation = ""
        if len(history) > 0:
            latest_usr_histoy, latest_ai_history = history[-2], history[-1]
            usr_role, ai_role = latest_usr_histoy["role"], latest_ai_history["role"]
            usr_chat, ai_chat = latest_usr_histoy["content"], latest_ai_history["content"]
            # conversation = usr_role + ": " + usr_chat + "\n" + ai_role + ": " + ai_chat + "\n"
            conversation = str({usr_role: usr_chat, ai_role: ai_chat})
            print(conversation)
        prompt = PROMPT_TEMPLATE_ZH["QUERY_TRANSFORM_TEMPLATE"].format(question=question, conversation=conversation)
        print(f"query transfer prompt:{prompt}")
        query_trans, _ = self.model.chat(self.tokenizer, question, prompt, history=[])
        return query_trans

class MiniCPM_Llama3_int4(BaseModel):
    def __init__(self, model_path, max_memory_map):
        super().__init__()
        self.model_path = model_path
        self.max_memory_map = max_memory_map
        self.load_model()

    def load_model(self):
        print("Start loading MiniCPM_Llama3 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, max_memory=self.max_memory_map, device_map = "auto")
        self.model.eval()
        print("Finish model building!")

    def chat_with_history(self, image, question, history, context = None):
        if context == None:
            # prompt = PROMPT_TEMPLATE["CHATGLM_TEMPLATE"].format(question=question)
            prompt = PROMPT_TEMPLATE_ZH["CPM_TEMPLATE"].format(question=question)
            print(prompt)
        else:
            prompt = PROMPT_TEMPLATE_ZH["RAG_CHATGLM_TEMPLATE"].format(question=question, text=context)
            print(prompt)
        history.append({'role': 'user', 'content': question})
        res = self.model.chat(image=image, msgs=history, tokenizer=self.tokenizer, sampling=True)
        history.append({"role": "assistant", "content": res})
        return res, history

    def chat(self, image, question, context = None):
        if context == None or context ==[]:
            prompt = PROMPT_TEMPLATE_ZH["CPM_TEMPLATE"].format(question=question)
            print("CPM Prompt:",prompt)
        else:
            prompt = PROMPT_TEMPLATE_ZH["RAG_CHATGLM_TEMPLATE"].format(question=question, text=context)
            print("CPM Prompt:",prompt)
        msgs = [{'role': 'user', 'content': question}]
        res = self.model.chat(image=image, msgs=msgs, tokenizer=self.tokenizer, sampling=True)
        return res


if __name__ == "__main__":
    model_dir = "chatglm3-6b"
    max_memory_map = {0: "10GB", 1: "10GB"}
    model = ChatGLM(model_dir, max_memory_map)

    # question = "simatic energy managerv7.5上市时间是多少？"
    question = "将上面的回答简短总结一下。"
    history = [{'role': 'user',
                'content': '详细介绍下s7-1200的基本功能。'
                },

               {'role':'assistant', 'metadata': '',
                'content': 'S7-1200是一款功能强大的控制器，具有集成电源和各种板载输入与输出电路。用户程序逻辑可以包含布尔逻辑、计数器、定时器和复杂数学运算。主要内容包括：\n\n1. 电源连接器：用于为CPU提供电源。\n2. 可拆卸用户接线连接器：用于连接外部设备。\n3. 板载I/O的状态LED：用于显示板载I/O的状态。\n4. PROFINET连接器（CPU的底部）：用于与编程设备通信，以及与HMI面板或其他CPU通信。\n\n此外，该表格还介绍了CPU的工作原理，但具体内容并未详细列出。', 'rag_img_path': None
                },
               ]
    query_trans = model.query_transform(question, history)
    print(query_trans)
    print(history)

    # rerank模型已下载完成