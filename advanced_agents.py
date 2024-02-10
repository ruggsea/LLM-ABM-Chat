from chat_llm import ReflectingAgent
from txtai import Embeddings
from llm_engines import LLMApi



class ReaActAgent(ReflectingAgent):
    def __init__(self, name:str, llm:LLMApi, prompt:str="", interests:list=[], behavior:list=[], n_last_messages:int=10, n_last_messages_reflection:int=20, n_examples:int=3):
        super().__init__(name, llm, prompt,interests, behavior, n_last_messages, n_last_messages_reflection)
        
    
        
        