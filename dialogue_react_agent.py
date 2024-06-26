from chat_llm import ReflectingAgent
from txtai import Embeddings
from llm_engines import LLMApi, LLM
import json, random, logging,re
from jinja2 import Template, Environment, FileSystemLoader



# loading the prompt from a file
def load_dialogue_react_fewshots(file_path:str):
    ## dialogue react prompts are contained in jsonl
    with open(file_path, "r") as f:
        prompts= [json.loads(line) for line in f]
    return prompts

def load_dialogue_act_taxonomy(file_path:str):
    ## dialogue act taxonomy is contained in txt file
    with open(file_path, "r") as f:
        acts = f.readlines()
    return [act.strip() for act in acts]

def load_base_prompt(file_path:str)->Template:
    ## the base prompt is saved inside a jinja template called dialogue_react_generation.j2 in the prompts folder
    # load the file
    with open(file_path, "r") as f:
        template = f.read()
    return Template(template)


def read_dialogue_act_list(file_path:str):
    ## the dialogue acts are saved in a json array inside dialogue_acts.json
    with open(file_path, "r") as f:
        dialogue_acts = json.load(f)
    return dialogue_acts

dialogue_acts_taxonomy=load_dialogue_act_taxonomy("prompts/dialogue_taxonomy.txt")
dialogue_react_fewshots = load_dialogue_react_fewshots("fewshots_mpc/generated_dialogue_reacts.jsonl")  
dialogue_react_fewshots_no_dialogue=load_dialogue_react_fewshots("fewshots_mpc/generated_dialogue_reacts_no_dialogue.jsonl")
# loading the prompts
prompt_template = load_base_prompt("prompts/dialogue_react_generation.j2")
prompt_template_no_dialogue_no_react = load_base_prompt("prompts/dialogue_react_generation_no_dialogue_no_react.j2")
prompt_template_no_dialogue = load_base_prompt("prompts/dialogue_react_generation_no_dialogue.j2")


memory_template = load_base_prompt("prompts/memory_generation.j2")
reflection_template = load_base_prompt("prompts/reflection_generation.j2")
dialogue_acts_list=read_dialogue_act_list("prompts/dialogue_acts.json")


logging.basicConfig(level=logging.INFO, filename="chat.log", filemode="w", format="%(asctime)-15s %(message)s")




class DialogueReactAgent(ReflectingAgent):
    def __init__(self, name:str, persona:str, llm:LLM=LLMApi(),memory_freq:int=25, reflections_freq:int=50, n_fewshots:int=5, ablation:str=None):
        """
        Initializes the DialogueReactAgent object.

        Args:
            name (str): The name of the agent.
            llm (LLM): The LLM object.
            persona (str): The persona of the agent.
            memory_freq (int): The frequency of memory updates (memories are generated every memory_freq turns).
            reflections_freq (int): The frequency of reflections (reflections are generated evert reflections_freq turns).
            n_fewshots (int): The number of few-shot examples.
            ablation (str): The ablation type. Should be one of None, "no_dialogue_no_react", "no_dialogue".

        Returns:
            None
        """
        self.name = name
        self.llm = llm
        self.persona = persona
        self.memory_freq = memory_freq
        self.reflections_freq = reflections_freq
        self.n_fewshots = n_fewshots
        self.memory = Embeddings(content=True, gpu=False)
        
        
        ## add the dialogue act taxonomy to the agent
        self.dialogue_acts_taxonomy = dialogue_acts_taxonomy
        ## dialogue act list 
        self.dialogue_acts = dialogue_acts_list

        # possible ablations
        ablation_types = ["no_dialogue_no_react", "no_dialogue"]
        if ablation != None and ablation not in ablation_types:
            assert ablation is None, f"Invalid ablation type: {ablation}."
        else:
            self.ablation = ablation    
            
        if self.ablation == None:
            self.prompt = prompt_template
        elif self.ablation == "no_dialogue_no_react":
            self.prompt = prompt_template_no_dialogue_no_react
        elif self.ablation == "no_dialogue":
            self.prompt = prompt_template_no_dialogue
        
    def gen_memories(self, last_messages, n_memories, **kwargs):
        """
        Generates a memory based on the last messages.

        Args:
            last_messages (list): The last messages.
            n_memories (int): The number of memories to generate.

        Returns:
            list of dicts: The generated memories.
        """
        
        # unpack kwargs
        agent_list = kwargs["agent_list"]
        n_agents = kwargs["n_agents"]
        turn_count = kwargs["turn_count"]
        
        # turn agent_list into a list of names
        agent_list = [agent.name for agent in agent_list]
        
        ## turn messages from tuples to dictionary
        
        last_messages = [{"turn_count":x[0], "sender":x[1], "message":x[2]} for x in last_messages]
        
        n_messages=len(last_messages)
        
        ## render the memory template
        memory_prompt = memory_template.render(name=self.name,
                                               persona=self.persona,
                                               last_messages=last_messages,
                                               n_messages=n_messages,
                                               n_memories=n_memories,
                                               n_agents=n_agents,
                                               agent_list=agent_list)
        
        
        # log compiled memory prompt
        logging.info(f"Memory prompt: {memory_prompt}")
        
        ## generate until you get the desired number of memories
        memories = []
        while len(memories)<n_memories:
            try:
                raw_response = self.llm.generate_response(memory_prompt)
                ## memories should be startin with a number and a dot and be separated by new lines
                memories_candidates = re.findall(r"\d+\..*", raw_response)
                for memory in memories_candidates:
                    # replace every number and dot with an empty string
                    memory = re.sub(r"\d+\.", "", memory)
                    memories.append(memory)
                    #print(f"Valid memory: {memory}")
            except Exception as e:
                print(f"Error generating response: {e}")
        
        # memories starting index
        memories_start_index=self.memory.count()
        
        # turn memories into dicts with turn_count and message
        memories_list = [{"memory_n":memories_start_index+i+1,"turn":turn_count, "text":memory, "type":"memory"} for i, memory in enumerate(memories)]
        
        return memories_list
             
    def save_observations(self, observations_list):
        
        """
        Saves observations in memory.

        Args:
            observations_list (list): The list of observations to save.
        """
        self.memory.upsert(observations_list)

    def gen_reflections(self, n_memories, n_reflections, **kwargs):
        
        """
        Generates a reflection based on the last memories.
        
        Args:
            n_memories (int): The number of memories to consider.
            n_reflections (int): The number of reflections to generate.
            
        Returns:
            list of dicts: The generated reflections.
        """
        
        agent_list = kwargs["agent_list"]
        n_agents = kwargs["n_agents"]
        turn_count = kwargs["turn_count"]
        
        # turn agent_list into a list of names
        agent_list = [agent.name for agent in agent_list]
        
        ## get the last n_last_memories
        
        last_memories = self.memory.search("select text, turn, memory_n from txtai order by memory_index desc", n_memories )
        
        reflection_prompt = reflection_template.render(name=self.name,
                                                       persona=self.persona,
                                                       last_memories=last_memories,
                                                       n_memories=n_memories,
                                                       n_reflections=n_reflections,
                                                       n_agents=n_agents,
                                                       agent_list=agent_list)
        
        # log compiled reflection prompt
        logging.info(f"Reflection prompt: {reflection_prompt}")
        
        ## generate until you get the desired number of reflections
        reflections = []
        while len(reflections)<n_reflections:
            try:
                raw_response = self.llm.generate_response(reflection_prompt)
                ## reflections should be starting with a number and a dot and be separated by new lines
                reflections_candidates = re.findall(r"\d+\..*", raw_response)
                for reflection in reflections_candidates:
                    # replace every number and dot with an empty string
                    reflection = re.sub(r"\d+\.", "", reflection)
                    reflections.append(reflection)
                    #print(f"Valid reflection: {reflection}")
            except Exception as e:
                print(f"Error generating response: {e}")
        
        
        # memories starting index
        reflections_start_index=self.memory.count()
        
        # turn memories into dicts with turn_count and message
        
        reflections_list = [{"memory_n":reflections_start_index+i+1,"turn":turn_count, "text":reflection, "type":"reflection"} for i, reflection in enumerate(reflections)]
        
        return reflections_list
                                                               
    def get_answer(self, last_messages, extra_context="", **kwargs):
    
        # unpack kwargs
        agent_list = kwargs["agent_list"]
        n_agents = kwargs["n_agents"]
        turn_count = kwargs["turn_count"]
        
        # if memory is empty, set optional_memory to ""
        if self.memory.isdense():
            last_message = last_messages[-1]
            optional_memory_list = self.memory.search(last_message[-1], limit=5)
            ## format the memory list
            optional_memory = "\n".join(x["text"] for x in optional_memory_list)
        else:
            optional_memory = ""
        
        
        dialogue_react_fewshots_sample=random.sample(dialogue_react_fewshots,1)[0]["generated_response"]
        
        
        # format the messages into a list of strings in the format "Agent: message"
        last_messages = [f"{x[1]}: {x[2]}" for x in last_messages]
        
        
        if self.ablation == None:
            # render template
            prompt = self.prompt.render(
                name=self.name,
                persona=self.persona,
                last_messages=last_messages,
                optional_memory=optional_memory,
                agent_list=agent_list,
                observation_thought_action_examples_list=dialogue_react_fewshots_sample,
                n_agents=n_agents,
                dialogue_acts_taxonomy=dialogue_acts_taxonomy
            )
        elif self.ablation == "no_dialogue_no_react":
            messages_pre_react=random.sample(dialogue_react_fewshots,1)[0]["messages"]
            # messages is an array containing text and speaker, it should be put in the format "Speaker: text\nSpeaker: text\n...Speaker: text\n"
            dialogue_react_fewshots_sample="\n".join([f"{x['speaker']}: {x['text']}" for x in messages_pre_react])
            # render template
            prompt = self.prompt.render(
                name=self.name,
                persona=self.persona,
                last_messages=last_messages,
                optional_memory=optional_memory,
                agent_list=agent_list,
                examples_list_no_dialogue_no_react=dialogue_react_fewshots_sample,
                n_agents=n_agents,
            )
        elif self.ablation == "no_dialogue":
            dialogue_react_fewshots_sample=random.sample(dialogue_react_fewshots_no_dialogue,1)[0]["generated_response"]
            prompt = self.prompt.render(
                name=self.name,
                persona=self.persona,
                last_messages=last_messages,
                optional_memory=optional_memory,
                agent_list=agent_list,
                observation_thought_action_examples_list=dialogue_react_fewshots_sample,
                n_agents=n_agents,
            )
        
        # log rendered prompt
        logging.info(f"Answer generation prompt: {prompt}")
        
        # get the answer for the dialogue react method
        if self.ablation == None:
            while True:
                ## the answer is generated by the llm and should be in the format
                ## Observation: <observation> 
                # Thought: <thought> 
                # Action: <action>
                try:
                    answer_candidate = self.llm.generate_response(prompt)
                    ## find the observation, thought and action
                    observation = re.findall(r"Observation: (.*?)\n", answer_candidate)
                    thought = re.findall(r"Thought: (.*?)\n", answer_candidate)
                    action = re.findall(r"Action: (.*?)\#\#", answer_candidate)
                    ## check that we have exactly one observation, thought and action
                    if len(observation)==1 and len(thought)==1 and len(action)==1:
                        ## check that thought is a dialogue act
                        observation = observation[0].strip()
                        thought = thought[0].strip()
                        action = action[0].strip()
                        if thought not in self.dialogue_acts:
                            logging.info(f"Invalid dialogue act: {thought}.")
                            continue
                        answer = action
                        #print(f"Valid answer: {answer_candidate}.")
                        logging.info(f"Valid dialogue_act: {thought}.")
                        logging.info(f"Valid answer: {answer}.")
                        break
                    else:
                        logging.info(f"Invalid answer: {answer_candidate}.")
                except Exception as e:
                    logging.info(f"Error generating response: {e}.")
                    logging.info(f"Invalid answer: {answer_candidate}.")
                    
            answer_with_dialogueAct = f"Following the observation: {observation}, I wanted to commit the following dialogue act: {thought}. Therefore, I wrote the message: {action}."                
            ## save the answer in memory
            self.memory.upsert([{"turn_count":turn_count, "text":answer_with_dialogueAct}])
        # generate answer for the ablation no_dialogue
        elif self.ablation == "no_dialogue":
            while True:
                ## the answer is generated by the llm and should be in the format
                ## Observation: <observation> 
                # Thought: <thought> 
                # Action: <action>
                try:
                    answer_candidate = self.llm.generate_response(prompt)
                    ## find the observation, thought and action
                    observation = re.findall(r"Observation: (.*?)\n", answer_candidate)
                    thought = re.findall(r"Thought: (.*?)\n", answer_candidate)
                    action = re.findall(r"Action: (.*?)\#\#", answer_candidate)
                    ## check that we have exactly one observation, thought and action
                    if len(observation)==1 and len(thought)==1 and len(action)==1:
                        # same as before, but without the dialogue act check
                        observation = observation[0].strip()
                        thought = thought[0].strip()
                        action = action[0].strip()
                        answer = action
                        logging.info(f"Valid answer: {answer}.")
                        break
                    else:
                        logging.info(f"Invalid answer: {answer_candidate}.")
                except Exception as e:
                    logging.info(f"Error generating response: {e}.")
                    logging.info(f"Invalid answer: {answer_candidate}.")
        # generate answer for the ablation no_dialogue_no_react
        elif self.ablation == "no_dialogue_no_react":
            # in this case, the message is a string that ends with ##
            while True:
                try:
                    answer_candidate = self.llm.generate_response(prompt)
                    if answer_candidate.endswith("##"):
                        answer = answer_candidate
                        # remove the ##
                        answer = answer[:-2]
                        logging.info(f"Valid answer: {answer}.")
                        break
                    else:
                        logging.info(f"Invalid answer: {answer_candidate}.")
                except Exception as e:
                    logging.info(f"Error generating response: {e}.")
                    logging.info(f"Invalid answer: {answer_candidate}.")
            ## save the answer in memory
            self.memory.upsert([{"turn_count":turn_count, "text":answer}])

        return answer
      
    def run_routines(self, turn_count, chat_history, agent_list, n_agents):
        """
        Runs routines for the agent. It will be called by the ChatThread object at the end of every turn.

        Args:
            turn_count (int): The current turn count.
            chat_history (list): The chat history.
        """
        if turn_count % self.memory_freq == 0:
            print(f"Time to generate memories for {self.name}!")
            self.save_observations(self.gen_memories(last_messages=chat_history[-self.memory_freq:], n_memories=5,agent_list=agent_list, n_agents=n_agents, turn_count=turn_count))
        if turn_count % self.reflections_freq == 0:
            print(f"Time to generate reflections for {self.name}!")
            self.save_observations(self.gen_reflections(n_memories=100, n_reflections=5, agent_list=agent_list, n_agents=n_agents, turn_count=turn_count))
        else:
            pass
    
    def dump_agent(self):
        """
        Dumps the agent to a file.
        """
        # dump the memory as a list of dictionaries
        memory_dump = self.memory.search("SELECT id, text, turn, n_memory, entry, type FROM txtai ORDER BY entry DESC", limit=10000)
        
        agent_data={
            "name":self.name,
            "persona":self.persona,
            "memory_freq":self.memory_freq,
            "reflections_freq":self.reflections_freq,
            "n_fewshots":self.n_fewshots,
            "memories":memory_dump,
            "llm": self.llm.__class__.__name__,
            "type": self.__class__.__name__
        }   

        return agent_data

def main():
    # Create an instance of the DialogueReactAgent
    agent = DialogueReactAgent("Agent", "A good person",LLMApi(), 0.5, 0.2, 5)

    # Perform some tests
    print(agent.llm)
    print(agent.persona)
    print(agent.memory_freq)
    print(agent.reflections_freq)
    print(agent.n_fewshots)
    print(agent.memory)
    print(agent.dialogue_acts[0])
    # agent.get_answer(["Hello", "How are you?"],
    #                  agent_list=["Agent1", "Agent2"],
    #                  n_agents=2,
    #                  turn_count=3)
    # test the ablation no_dialogue
    agent=DialogueReactAgent("Agent", "A good person that likes golf",LLMApi(), 0.5, 0.2, 5, "no_dialogue")
    answer = agent.get_answer([("Agent1", "Hello", "How are you?")],
                     agent_list=["Agent1", "Agent2"],
                     n_agents=2,
                     turn_count=3)
    print(answer)
if __name__ == "__main__":
    main()
        
        
        

        
    