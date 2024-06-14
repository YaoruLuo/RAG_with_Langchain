from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ChatGLM4_LLM(LLM):
    # 基于本地 ChatGLM4 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    gen_kwargs: dict = None

    def __init__(self, mode_name_or_path: str, gen_kwargs: dict = None):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda:2",
            # max_memory = {0:"20GB", 1:"20GB", 2:"20GB"}
        ).to("cuda").eval()
        print("完成本地模型的加载")

        if gen_kwargs is None:
            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        self.gen_kwargs = gen_kwargs

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        model_inputs = model_inputs.to('cuda')
        generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别LLM的字典,这对于缓存和跟踪目的至关重要。"""
        return {
            "model_name": "glm-4-9b-chat",
            "max_length": self.gen_kwargs.get("max_length"),
            "do_sample": self.gen_kwargs.get("do_sample"),
            "top_k": self.gen_kwargs.get("top_k"),
        }

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"

if __name__ == "__main__":

    gen_kwargs = {"max_length": 2500}
    llm = ChatGLM4_LLM(mode_name_or_path="glm-4-9b-chat", gen_kwargs=gen_kwargs)
    print(llm.invoke("你是谁"))
