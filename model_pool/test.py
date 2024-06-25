from typing import Optional, List, Dict, Any
from langchain import LLM, CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_pool.promptTemplate import PROMPT_TEMPLATE_ZH, PROMPT_TEMPLATE_ZH_LC
import torch


class ChatGLM4_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    gen_kwargs: dict = None

    def __init__(self, mode_path: str, gpu_device, gen_kwargs: dict = None):
        super().__init__()
        print("Loading GLM4...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=gpu_device,
        ).to("cuda").eval()
        print("Finish loading GLM4!")

        self.gen_kwargs = gen_kwargs or {}

    def _generate_response(self, messages):
        """Generate response given a list of messages."""
        print(f"====input messages for model====: {messages} \n")

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )

        model_inputs = model_inputs.to('cuda')
        generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:

        chat_history = kwargs.get('chat_history', [])
        context = kwargs.get('context')
        prompt_template = kwargs.get('prompt_template')

        if prompt_template == "CHATGLM_TEMPLATE":
            system_prompt = PROMPT_TEMPLATE_ZH_LC[prompt_template]
        elif prompt_template == "RAG_CHATGLM_TEMPLATE":
            system_prompt = PROMPT_TEMPLATE_ZH_LC[prompt_template].format(context=context)
        elif prompt_template == "QUERY_TRANSFORM_TEMPLATE":
            system_prompt = PROMPT_TEMPLATE_ZH_LC[prompt_template]

        if not chat_history:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            return self._generate_response(messages)

        else:
            messages = chat_history + [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self._generate_response(messages)
            chat_history.pop(-1)
            chat_history.pop(-1)
            return response


    def query_transfer(self, question, history) -> str:
        conversation = ""
        if len(history) > 0:
            latest_usr_history, latest_ai_history = history[-2], history[-1]
            conversation = str({latest_usr_history["role"]: latest_usr_history["content"],
                                latest_ai_history["role"]: latest_ai_history["content"]})

        system_prompt = PROMPT_TEMPLATE_ZH_LC["QUERY_TRANSFORM_TEMPLATE"].format(conversation=conversation)

        print(f"====prompt query transfer====:{system_prompt} \n")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        return self._generate_response(messages)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for caching and tracking purposes."""
        return {
            "model_name": "glm-4-9b-chat",
            "max_length": self.gen_kwargs.get("max_length", None),
            "do_sample": self.gen_kwargs.get("do_sample", None),
            "top_k": self.gen_kwargs.get("top_k", None),
        }

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"


def build_chat_history(history, question, answer, rag_img_path = None):
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "rag_img_path": rag_img_path}
         ]
    )
    return history

if __name__ == "__main__":

    gen_kwargs = {"max_length": 2500}
    model_dir = "glm-4-9b-chat"
    gpu_device = "cuda:2"
    # prompt_template = "CHATGLM_TEMPLATE"
    prompt_template = "RAG_CHATGLM_TEMPLATE"

    llm = ChatGLM4_LLM(mode_path=model_dir,
                       gpu_device=gpu_device,
                       gen_kwargs=gen_kwargs)


    history = []

    question1 = "which company build the PLC?"
    context1 = "PLC build by simense."

    print("query transfer: ",llm.query_transfer(question1,history))


    answer1 = llm.invoke(question1,
                         chat_history=history,
                         context=context1,
                         prompt_template=prompt_template)

    history = build_chat_history(history, question1, answer1)
    print(f"answer1: {answer1} \n ============ \nhistory1: {history}")

    print("===================")

    question2 = "introduce yourself"
    context2 = ""

    answer2 = llm.invoke(question2,
                         chat_history=history,
                         context=context2,
                         prompt_template=prompt_template)

    history = build_chat_history(history, question2, answer2)
    print(f"chat2: {answer2} \n ============ \nhistory2: {history}")