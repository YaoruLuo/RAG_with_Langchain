import json5
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import json
import requests
from chatApp.model_pool.LLM import ChatGLM


tools = [
    {
        "name": "track",
        "description": "追踪指定股票的实时价格",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "description": "需要追踪的股票代码"
                }
            },
            "required": ['symbol']
        }
    },
    {
        "name": "text-to-speech",
        "description": "将文本转换为语音",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "description": "需要转换成语音的文本"
                },
                "voice": {
                    "description": "要使用的语音类型（男声、女声等）"
                },
                "speed": {
                    "description": "语音的速度（快、中等、慢等）"
                }
            },
            "required": ['text']
        }
    },
    {
        "name": "google_search",
        "description": "Google Search is a general search engine.",
        "parameters":{
            "type": "object",
            "properties":{
                "query":{
                    "description": "需要搜索的问题"
                }
            },
            "required": ['query']
        }
    }
]


def google_search(query: str):
    print("wwwwww")

    return response

function_map = {
    "google_search": google_search
}

class BaseModel:

    def chat(self, prompt, history, content):
        pass

    def load_model(self):
        pass



if __name__ == "__main__":
    model_dir = "../model_pool/chatglm3-6b"
    max_memory_map = {0: "11GB", 1: "11GB"}
    model = ChatGLM(model_dir, max_memory_map)

    system_info = {"role": "system",
                   "content": "Answer the following questions as best as you can. You have access to the following tools:",
                   "tools": tools}
    history = [system_info]

    # question = "simatic energy managerv7.5上市时间是多少？"
    question = "罗曜儒是谁？"

    response, history = model.chat(question, history)
    print("response:",response)
    print('=' * 50)
    print("history:", history)

    if isinstance(response, dict):
        func = function_map[response['name']]
        param = response["parameters"]
        func_response = func(**param)
        results = json5.dumps(func_response, ensure_ascii=False)
        response, history = model.chat(results, history=history, role = "observation")

        print("response:", response)
        print('=' * 50)
        print("history:", history)

