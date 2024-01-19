# implementingc llm model class that gives completions using openai-like api


import requests

url= "http://127.0.0.1:1200/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}



# first a dummy llm model class that just gives dummy responses
class LLM:
    instance = None
    def __init__(self, history=[]):
        self.headers = headers
        self.history = history
        # get first prompt with role system as system prompt
        if len(self.history) > 0:
            self.system_prompt = self.history[0]["content"] if self.history[0]["role"] == "system" else ""
        else:
            self.system_prompt = ""
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        if len(self.history) > 0:
            if self.history[0]["role"] == "system":
                self.history[0]["content"] = system_prompt
            else:
                self.history.insert(0, {"role":"system","content": system_prompt})
        else:
            self.history.insert(0, {"role":"system","content": system_prompt})
    
    def get_system_prompt(self):
        return self.system_prompt
    
    def set_history(self, history):
        self.history = history
        # update system prompt
        if len(self.history) > 0:
            self.system_prompt = self.history[0]["content"] if self.history[0]["role"] == "system" else ""
        else:
            self.system_prompt = ""
    
    def get_history(self):
        return self.history
    
    def generate_response(self, user_prompt):
        # append user prompt to history
        self.history.append({"role":"user","content": user_prompt})
        # generate dummy response
        response = "dummy response"
        # append response to history
        self.history.append({"role":"assistant","content": response})
        return response
    
    
# now a real llm model class that gives completions using openai-like api

class LLMApi(LLM):
    def __init__(self, history=[]):
        super().__init__(history)
        self.url = url
        self.headers = headers
    
    def generate_response(self, user_prompt):
        
        # append user prompt to history
        self.history.append({"role":"user","content": user_prompt})
        # query api for response
        data = {
        "mode": "instruct",
        "messages": self.history
        }
        response = requests.post(url, headers=headers, json=data, verify=False)
        
        assistant_message = response.json()['choices'][0]['message']['content']
        self.history.append({"role": "assistant", "content": assistant_message})
        
        
        
        
        return assistant_message


        
        