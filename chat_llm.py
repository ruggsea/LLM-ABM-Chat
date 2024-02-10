
from llm_engines import LLMApi
import random, textwrap, logging, requests, json, os, time, re
from txtai import Embeddings



logging.basicConfig(level=logging.INFO, filename="chat.log", filemode="w", format="%(asctime)-15s %(message)s")



basic_answer_generation_prompt = """Send a message to the group. Keep your answers short and casual and do not sound too excited but try to keep the conversationg going as much as possible. Be creative and engaging. 
Provide your answer as you would text it. Do not send links, images and videos in your messages and don't sign your messages. Do not use markdown formatting.Avoid greeting people and saying hi if you have already greeted them before. Also, avoid saying bye because we don't want the chat to end. Each message should be shorter than 500 characters\n"""


react_basic_answer_generation_prompt = """Come up with a message to send to the group by first generating an observation and a thought. Keep your answers short and casual and do not sound too excited but try to keep the conversationg going as much as possible. Be creative and engaging. 
Do not send links, images and videos in your messages and don't sign your messages. Do not use markdown formatting.Avoid greeting people and saying hi if you have already greeted them before
Keep your message answers short and casual and do not sound too excited but try to keep the conversationg going as much as possible. Be creative and engaging. Also, avoid saying bye because we don't want the chat to end. Each message should be shorter than 500 characters\n"""




eval_prompt="""
You are a chat evaluator that specializes in evaluating the naturalness of chat conversations with harsh but fair ratings. Y
You are evaluating a chat. You are evaluating the chat to see if this chat resembles a natural and believable conversation.
The chat history is:
###
{history_str}
###
How do you evaluate the chat from 1 to 10? 1 being the worst and 10 being the best. Answer with your reasoning, followed by a newline and the final evaluation. End your answer with ##. So the format is:

reason.\n number ##

Answer:
"""



eval_prompt="""
You are a chat evaluator that specializes in evaluating the naturalness and believability of chat conversations with harsh but fair ratings. Your expertise is in assessing conversations using specific criteria to determine how closely they resemble a natural and believable human conversation. For this task, you will focus on the following criteria:

1. Natural Language Understanding (NLU) - Does the chat demonstrate a clear understanding of the nuances of human language, including slang, idioms, and colloquial expressions?
2. Contextual Relevance - Does the chat maintain coherence and context throughout the conversation?
3. Emotional Intelligence - Does the chat detect and respond appropriately to the emotional content of the conversation?
4. Conversational Flow - Does the conversation flow smoothly without awkward pauses or abrupt topic changes?
5. Adaptability - Can the chat adapt to different styles of communication and handle unexpected inputs gracefully?

Please evaluate the following chat history:

####
{history_str}
####

For each of the criteria above, provide a brief assessment. Then, give an overall rating based on your analysis of these criteria from 1 to 10, where 1 is the worst and 10 is the. Your response should follow this format:

1. NLU: Reasoning.
2. Contextual Relevance: Reasoning. 
3. Emotional Intelligence: Reasoning 
4. Conversational Flow: Reasoning.
5. Adaptability: Reasoning.

Final Evaluation Reasoning.
Final Rating: 
number/10##

End your answer with ##.

Answer:
"""


# caching the messages dataset
if os.path.exists("example_messages.json"):
    with open("example_messages.json", "r") as f:
        example_messages = json.load(f)
else:
    chat_data_url="https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/train.json"
    data=requests.get(chat_data_url).json()
    example_messages=list()
    # extract content message from the json
    for k in data.keys():
        # get content
        content=data[k]["content"]
        # for each message in the content
        for message in content:
            # print the message
            example_messages.append(message["message"])
    with open("example_messages.json", "w") as f:
        json.dump(example_messages, f)


# caching the react examples
with open("react_examples.json", "r") as f:
    react_examples = json.load(f)

class Agent:
    def __init__(self, name, llm, prompt="", interests=[], behavior=[], n_examples=3, react=True): 
        """
        Initializes an Agent object.

        Args:
            name (str): The name of the agent.
            llm (LLMApi): The LLM engine, it defaults to LLMAPI.
            prompt (str, optional): The initial prompt for the agent. If left empty, a modular prompt is generated using the interests and behavior arguments. Defaults to an empty string.
            interests (list, optional): The interests of the agent. Defaults to an empty list.
            behavior (list, optional): The behavior of the agent. Defaults to an empty list.
            n_examples (int, optional): The number of examples to use in the agent's answer generation. Defaults to 3.
            react (bool, optional): Whether the few shot examples should be ReAct examples. Defaults to True.
            
        """
        self.name = name
        
        self.interests = interests
        self.behavior = behavior
        
        if prompt == "":
            interests_str = ", ".join(interests)
            behavior_str = ", ".join(behavior)
            self.prompt = f"""
You are a person called {self.name}. You are texting with some friends in a telegram group. 
You have interests in {interests_str}. You usually show the following attitude: {behavior_str}. 
            """
        else:
            self.prompt = prompt
            
        
        self.llm = llm
        self.llm.set_system_prompt(self.prompt)
        self.agent_answers = []
        self.n_examples = n_examples
        self.react = react
    
    def get_name(self):
        """
        Returns the name of the agent.

        Returns:
            str: The name of the agent.
        """
        return self.name
    
    
    def log(self, prompt):
        """
        Logs a prompt for debugging answer generation.

        Args:
            prompt (str): The prompt to log.
        """
        logging.info(prompt)
    

    
    def get_answer(self, last_messages, extra_context=""):
        """
        Generates an answer from the agent given the last messages in the chat.

        Args:
            last_messages (list): The list of previous messages in the chat.
            extra_context (str, optional): Additional context for the agent's answer. Defaults to an empty string.

        Returns:
            str: The agent's answer.
        """
        formatted_last_messages_str = []
        last_messages = last_messages[-5:] # Only consider the last 5 messages  
        for message_n, agent, message in last_messages:
            formatted_last_messages_str.append(f"{message_n}. {agent}: {message}")
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        
        if self.react:
            few_shot_examples = random.sample(react_examples, self.n_examples)
            format='"Observation: Thought: Action: message ##"'
            
            # add quotes around the examples
            few_shot_examples = [f"'{x}'" for x in few_shot_examples]
            
            # join the examples into a string with two new lines between each example
            few_shot_examples_str = "\n\n".join(few_shot_examples)
        else:
            few_shot_examples = random.sample(example_messages, self.n_examples)
            format='"message ##"'
            # add quotes around the examples
            few_shot_examples = [f"'{x}'" for x in few_shot_examples]
            # add ## to the end of each example
            few_shot_examples = [f"{x} ##" for x in few_shot_examples]
            # join the examples into a string
            few_shot_examples_str = "\n".join(few_shot_examples)

        
        
        seek_answer_prompt = f"""
{basic_answer_generation_prompt if self.react==False else react_basic_answer_generation_prompt}
The last messages (they are shown in the format "message_number. name: message") were:
###
{formatted_last_messages_str}
###

How do you answer? Give your answer {"as you would text it," if self.react==False else ""} in the format {format}. End your answer with ##.  Some examples:

{few_shot_examples_str}

{"'message ##'" if self.react==False else ""}

{"Your answer:" if self.react==False else ""}\n
###
{self.name+":" if self.react==False else ""}"""
        
        if extra_context != "":
            seek_answer_prompt = f"""{self.prompt}
{extra_context}
{seek_answer_prompt}"""
        else:
            seek_answer_prompt = f"""{self.prompt}
{seek_answer_prompt}"""
        
        agent_answer = ""
        
        self.log(seek_answer_prompt)
        
        while len(agent_answer) < 5:
            agent_answer = self.llm.generate_response(seek_answer_prompt)
            if agent_answer == "":
                print("Empty answer returned. Retrying...")
            if self.react:
                # find the string between Action: and ## using regex
                pattern = r"Action: (.*?)##"
                matches = re.findall(pattern, agent_answer)
                if matches:
                    agent_answer = matches[-1]
                    agent_answer = agent_answer + "##"
                else:
                    print("Invalid answer: no React style answer found. Retrying...")
                    logging.info(f"Invalid answer: {agent_answer}.")
                    agent_answer = ""
                    continue
        
        self.agent_answers.append(agent_answer)
        return agent_answer
    
    def run_routines(self, turn_count, chat_history):
        """
        Runs routines for the agent. It will be called by the ChatThread object at the end of every turn.

        Args:
            turn_count (int): The current turn count.
            chat_history (list): The chat history.
        """
        pass
    
    # agent will need to be dumped from the chat thread
    def dump_agent(self):
        """
        Turn the agent into a dictionary for dumping into a json file.
        
        Returns:
            dict: The agent's data.
        """
        agent_data = {"name": self.name, "prompt": self.prompt, "agent_answers": self.agent_answers , "type":self.__class__.__name__, "n_examples": self.n_examples, "interests": self.interests, "behavior": self.behavior, "llm": self.llm.__class__.__name__}
        return agent_data


class ChatThread:
    def __init__(self, agent_list=[], chat_goal="There is not a specific goal for this conversation.", neutral_llm=LLMApi(), sel_method="random", n_eval=10):
        """
        Initializes a ChatThread object.

        Args:
            agent_list (list, optional): The list of agents participating in the chat. Defaults to an empty list.
            chat_goal (str, optional): The goal of the chat. Defaults to "There is not a specific goal for this conversation.".
            neutral_llm (LLMApi, optional): The language model API for LLM generation needs outside the scope of any agent. Defaults to LLMApi().
            sel_method (str, optional): The method for selecting the next agent to answer. Defaults to "random".
            n_eval (int, optional): The number of turns between evaluations. Defaults to 10. Put it to -1 to disable evaluations.
        """
        self.chat_history = []
        self.agent_list = agent_list
        self.chat_goal = chat_goal
        self.n_eval = n_eval
        self.conversation_starters = ["Hi!", "Hello!", "How are you?", "How is it going?", "What's up?", "How are you doing?"]
        self.neutral_llm = neutral_llm
        self.sel_method = sel_method
        
        # create an attribute to store chat data for then dumping it into a json file
        self.chat_evaluation = {}
        
        # create a unique identifier for the chat using the current time
        self.chat_id =f"chat_{str(int(time.time()))}"
        
        
    def pick_random_agent(self):
        """
        Picks a random agent from the agent list.

        Returns:
            Agent: The randomly selected agent.
        """
        random_agent = random.choice(self.agent_list)
        return random_agent
    
    def start_conversation(self):
        """
        Starts a conversation by picking a random agent and a random conversation starter.
        
        Returns:
            tuple: The start message of the conversation.
        """
        assert len(self.chat_history) == 0
        
        random_agent = self.pick_random_agent()
        random_conversation_starter = random.choice(self.conversation_starters)
        start_message = (1,random_agent.get_name(), random_conversation_starter)
        self.chat_history.append(start_message)
        return start_message
    
    def get_chat_answer(self, last_messages, agent):
        """
        Gets an answer from an agent.

        Args:
            last_messages (list): The list of last messages in the chat.
            agent (Agent): The agent providing the answer.

        Returns:
            str: The agent's answer.
        """
        other_agents = self.agent_list.copy()
        other_agents.remove(agent)
    
        other_agents_str = ", ".join([agent.get_name() for agent in other_agents])
        
        extra_context = f"""
In this chat, there's you ({agent.get_name()}) and {other_agents_str}.
The converation should be implicitly steered towards the following goal:
{self.chat_goal}. 
"""
        # check if the answer fits the validation criteria
        
        validation_ending="##"
        agent_answer = ""
        while not agent_answer.endswith(validation_ending):
            agent_answer = agent.get_answer(last_messages=last_messages, extra_context=extra_context)
             # trim the answer to remove everything after the validation ending
            ending_index = agent_answer.find(validation_ending)
            if ending_index == -1:
                logging.info("Invalid answer. Retrying...")
                logging.info(f"Invalid answer: {agent_answer}.")
                continue
            else:
                agent_answer = agent_answer[:ending_index]+ validation_ending
            
        # remove the validation ending
        agent_answer = agent_answer[:agent_answer.find(validation_ending)]
        self.chat_history.append((self.turn_count,agent.get_name(), agent_answer))
        return agent_answer
    
    def render_last_message(self):
        """
        Renders the last message in the chat history.
        """
        last_message = self.chat_history[-1]
        n_message, agent, message =  last_message[0], last_message[1], last_message[2]
        
        msg_string = f"\033[91m{agent}\033[0m: {message}"
        wrapped_message = textwrap.fill(msg_string, width=80, subsequent_indent=' ' * 4)
        print(wrapped_message)
    
    
    def evaluate_chat(self, start_index=0, end_index=-1):
        """
        Evaluates the chat history using the neutral llm
        
        Parameters:
        start_index (int): The index of the chat history to start the evaluation from. Defaults to 0.
        end_index (int): The index of the chat history to end the evaluation. Defaults to -1.
        
        Returns:
            dict: The evaluation of the chat.
        """
        
        
        history_str = "\n".join([f"{x[1]}: {x[2]}" for x in self.chat_history[start_index:end_index]])
        evaluation_prompt = eval_prompt.format(history_str=history_str)
        evaluation_answer = ""
        evaluation_count = 3
        total_evaluation = 0

        logging.info(evaluation_prompt)
        
        
        # a re pattern to find "number/10" in the answer
        pattern = r"\b(?:10|\d(?:\.\d+)?)\/10\b"
        
        
        for _ in range(evaluation_count):
            # check if the answer is a valid number
            while not isinstance(evaluation_answer, float):
                evaluation_answer = self.neutral_llm.generate_response(evaluation_prompt)
                # try to trim the answer to remove everything after the validation ending
                ending_index = evaluation_answer.find("##")
                logging.info(f"Answer: {evaluation_answer}")
                if ending_index == -1:
                    logging.info("Invalid answer: no ## at the end. Retrying...")
                    continue
                else:
                    evaluation_answer = evaluation_answer[:ending_index]
                try:
                    answer_candidate = evaluation_answer
                    
                    # find "number/10" pattern in the answer
                    matches = re.findall(pattern, evaluation_answer)
                    if matches:
                        rating = matches[-1]
                        rating = rating.split("/")[0]
                        evaluation_answer = float(rating)
                    else:
                        logging.info("Invalid answer: no rating found. Retrying...")
                        continue
                except:
                    logging.info("Invalid answer: error while trying to find the rating. Retrying...")
                    continue

            total_evaluation += evaluation_answer
            evaluation_answer = ""

        average_evaluation = total_evaluation / evaluation_count
        return average_evaluation
        
        
    def dump_chat(self):
        """
        Dumps the chat history into a json file.
        """
        chat_data = {"chat_id": self.chat_id, "chat_history": self.chat_history, "chat_evaluation": self.chat_evaluation, "chat_goal": self.chat_goal, "agent_list": [agent.dump_agent() for agent in self.agent_list], "neutral_llm": self.neutral_llm.__class__.__name__, "sel_method": self.sel_method, "n_eval": self.n_eval}
        if not os.path.exists("chat_logs"):
            os.makedirs("chat_logs")
        with open(f"chat_logs/{self.chat_id}.json", "w") as f:
            json.dump(chat_data, f)    
    
    
    def run_chat(self, max_turns=50):
        """
        Runs a chat simulation for a maximum number of turns. As of now, the simulations uses a random agent selection method.

        Args:
            max_turns (int, optional): The maximum number of turns in the chat. Defaults to 50.

        Returns:
            list: The chat history.
        """
        self.start_conversation()
        self.render_last_message() 

        starting_agent_name = self.chat_history[0][1]
        starting_agent = [agent for agent in self.agent_list if agent.get_name() == starting_agent_name][0]
        prev_agent = starting_agent
        
        global turn_count
        self.turn_count = 1
        turn_count = self.turn_count
        
        
        while self.turn_count < max_turns:
            for agent in self.agent_list:
                agent.run_routines(self.turn_count, self.chat_history)
                
            # every n turns evaluate the chat
            if self.turn_count % self.n_eval == 0 and self.n_eval != -1:
                # evaluate the chat (last n messages)
                step_evaluation = self.evaluate_chat(start_index=self.turn_count-self.n_eval, end_index=self.turn_count)
                print(f"The last chunck of the chat has been evaluated as {step_evaluation}/10.")
                
                evaluation = self.evaluate_chat()
                print(f"The overall chat has been evaluated as {evaluation}/10.")
            
                # save the evaluation in the chat evaluation attribute
                self.chat_evaluation[self.turn_count] = {"evaluation": evaluation, "step_evaluation": step_evaluation}
                
                
            
            if self.sel_method == "random":
                random_agent = self.pick_random_agent()
                while random_agent == prev_agent:
                    random_agent = self.pick_random_agent()

            last_messages = self.chat_history
            # update turn count and get answer for the turn
            self.turn_count += 1
            turn_count = self.turn_count
            self.get_chat_answer(last_messages, random_agent)
            self.render_last_message()
            prev_agent = random_agent
            
        # at the end of the chat, evaluate the conversation
        if self.n_eval != -1:
            evaluation = self.evaluate_chat()
            step_evaluation = self.evaluate_chat(start_index=self.turn_count-self.n_eval, end_index=self.turn_count)
            print(f"The chat has been evaluated as {evaluation}/10.")
            
            # save the evaluation in the chat evaluation attribute
            self.chat_evaluation[self.turn_count] = {"evaluation": evaluation, "step_evaluation": step_evaluation}
            
            
        # save the chat data into a json file
        self.dump_chat()
        
        return self.chat_history
    





class MemoryAgent(Agent):
    def __init__(self, name, llm, prompt="", interests=[], behavior=[], n_last_messages=10, n_examples=3, react=True):
        """
        Initializes a Agent object with memory.

        Args:
            name (str): The name of the agent.
            llm (LLMApi): The language model API.
            prompt (str, optional): The initial prompt for the agent. Defaults to an empty string.
            interests (list, optional): The interests of the agent. Defaults to an empty list.
            behavior (list, optional): The behavior of the agent. Defaults to an empty list.
            n_last_messages (int, optional): The number of messages that gets considered when generating memories. Defaults to 10.
            n_examples (int, optional): The number of examples to use in the agent's answer generation. Defaults to 3.
            react (bool, optional): Whether the few shot examples should be ReAct examples. Defaults to True.
            
        """
        super().__init__(name, llm, prompt, interests, behavior, n_examples ,react)
        self.memory = Embeddings(content=True, gpu=False)
        self.n_last_messages = n_last_messages
        
        
    def get_observations(self, last_messages):
        """
        Gets observations from the last messages.

        Args:
            last_messages (list): The list of last messages in the chat.

        Returns:
            list: The observations.
        """
        formatted_last_messages_str = []
        print(f"Processing observations for {self.name}...")

        for message_n, agent, message in last_messages[-self.n_last_messages:]:
            formatted_last_messages_str.append(f"{message_n}. {agent}: {message}")
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        observation_prompt = f"""
Your goal is to remember what happened in the last messages by saving memorable observations in your memory.
Those are the last messages in the chat (they are shown in the format "message_number. name: message"):
###
{formatted_last_messages_str}
###
Summarize the last messages in the chat in 5 observations, divided by a new line. The observations should be short and casual., useful for remembering what happened in the chat and should strictly be a truthful accounts of what is talked about in the chat.
Use this format for your observations:

1. Observation 1 \n
2. Observation 2 \n
3. Observation 3 \n
etc.
    
        """

        observation_prompt_final = f"""
{self.prompt}\n
{observation_prompt}
        """
        
        
        observations_list = []
        # log the prompt for debugging
        self.log(observation_prompt_final)
        while len(observations_list) < 5:
            try:
                observations = self.llm.generate_response(observation_prompt_final)
                observations_list = observations.split("\n")
            except Exception as e:
                print(f"Observations could not be generated: {e}. Retrying...")
                continue
            if len(observations_list) < 5:
                print("Not enough observations generated. Retrying...")
                continue
            # give obserations the turn count
            observations_list = [{"text": x, "turn": turn_count} for x in observations_list]

        return observations_list
    
    def save_observations(self, observations_list):
        """
        Saves observations in memory.

        Args:
            observations_list (list): The list of observations to save.
        """
        self.memory.upsert(observations_list)
        
    def get_answer(self, last_messages, extra_context=""):
        """
        Generates an answer from the agent enriching the context with observations from memory.

        Args:
            last_messages (list): The list of last messages in the chat.
            extra_context (str, optional): Additional context for the agent's answer. Defaults to an empty string.

        Returns:
            str: The agent's answer.
        """
        formatted_last_messages_str = []
        
        for message_n, agent, message in last_messages[-5:]:
            message = message.replace("\n", " ")
            formatted_last_messages_str.append(f"{message_n}. {agent}: {message}")
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        
        
        if self.react:
            few_shot_examples = random.sample(react_examples, self.n_examples)
            format='"Observation: Thought: Action: message ##"'
            
            # add quotes around the examples
            few_shot_examples = [f"'{x}'" for x in few_shot_examples]
            
            # join the examples into a string with two new lines between each example
            few_shot_examples_str = "\n\n".join(few_shot_examples)
        else:
            few_shot_examples = random.sample(example_messages, self.n_examples)
            format='"message ##"'
            # add quotes around the examples
            few_shot_examples = [f"'{x}'" for x in few_shot_examples]
            # add ## to the end of each example
            few_shot_examples = [f"{x} ##" for x in few_shot_examples]
            # join the examples into a string
            few_shot_examples_str = "\n".join(few_shot_examples)

        
        seek_answer_prompt = f"""
{basic_answer_generation_prompt if self.react==False else react_basic_answer_generation_prompt}
The last messages in the chat were (they are shown in the format "message_number. name: message"): 
###
{formatted_last_messages_str}
###

How do you answer? Give your answer {"as you would text it," if self.react==False else ""} in the format {format}. End your answer with ##.  Some examples:

{few_shot_examples_str}

{"'message ##'" if self.react==False else ""}

Your answer:          
###
{self.name if self.react==False else ""}
"""
        
        if self.memory.isdense():   
            last_message = last_messages[-1]
            memories = self.memory.search(last_message[-1], limit=5)
            context = "\n".join(x["text"] for x in memories)
            if extra_context != "":
                seek_answer_prompt_with_context = f"""
{self.prompt}\n
{extra_context}\n
{seek_answer_prompt}
The last messages in the chat were (they are shown in the format "message_number. name: message"): 
###
{formatted_last_messages_str}
###
How do you answer? Give your answer {"as you would text it," if self.react==True else ""} in the format {format}. End your answer with ##.  Some examples:

{few_shot_examples_str}

{"'message ##'" if self.react==False else ""}

In addition, you remember the following memories relevant to chat right now:
{context}\n

Your answer:          
###
{self.name}:
"""
            else:
                seek_answer_prompt_with_context = f"""
{self.prompt}\n
{basic_answer_generation_prompt if self.react==False else react_basic_answer_generation_prompt}
The last messages in the chat were (they are shown in the format "message_number. name: message"): 
###
{formatted_last_messages_str}
###

How do you answer? Give your answer {"as you would text it," if self.react==False else ""} in the format {format}. End your answer with ##.  Some examples:

{few_shot_examples_str}
{"'message ##'" if self.react==False else ""}
                
In addition, you remember the following memories relevant to chat right now:
{context}\n

Your answer:          
###
{self.name}:
"""
        else:
            if extra_context != "":
                seek_answer_prompt_with_context = f"""
{self.prompt}\n
{extra_context}\n
{seek_answer_prompt}
"""
            else:
                seek_answer_prompt_with_context = f"""
{self.prompt}\n
{seek_answer_prompt}            
"""

        agent_answer = ""
        
        self.log(seek_answer_prompt_with_context)

        while len(agent_answer) < 5:
            agent_answer = self.llm.generate_response(seek_answer_prompt_with_context)
            if agent_answer == "":
                print("Empty answer returned. Retrying...")
            if self.react:
                # find the string between Action: and ## using regex
                pattern = r"Action: (.*?)##"
                matches = re.findall(pattern, agent_answer)
                if matches:
                    agent_answer = matches[-1]
                    # readd the validation ending
                    agent_answer = agent_answer + "##"
                else:
                    print("Invalid answer: no React style answer found. Retrying...")
                    logging.info(f"Invalid answer: {agent_answer}.")
                    agent_answer = ""
                    continue
        
        self.agent_answers.append(agent_answer)
        return agent_answer
    
    def run_routines(self, turn_count, chat_history):
        """
        Runs routines for the agent. It will be called by the ChatThread object at the end of every turn.

        Args:
            turn_count (int): The current turn count.
            chat_history (list): The chat history.
        """
        if turn_count % self.n_last_messages == 0:
            print(f"Time to generate memories for {self.name}!")
            self.save_observations(self.get_observations(chat_history))
        else:
            pass
    def dump_agent(self):
        """
        Turn the agent into a dictionary for dumping into a json file.
        
        Returns:
            dict: The agent's data.
        """
        # dump the memory as a list of dictionaries
        memory_dump = self.memory.search("SELECT id, text, turn, entry FROM txtai ORDER BY entry DESC", limit=100)
        
        
        agent_data = {"name": self.name, "prompt": self.prompt, "agent_answers": self.agent_answers, "memories": memory_dump, "type":self.__class__.__name__, "n_last_messages": self.n_last_messages, "n_examples": self.n_examples, "interests": self.interests, "behavior": self.behavior, "llm": self.llm.__class__.__name__}
        return agent_data

    


class ReflectingAgent(MemoryAgent):
    def __init__(self, name, llm, prompt="", interests=[], behavior=[], n_last_messages=10, n_last_messages_reflection=20, n_examples=3, react=True):
        """
        Initializes a ReflectingAgent object, an agent that creates reflections from memories every n_last_messages_reflection turns.
        
        Args:
        name (str): The name of the agent.
        llm (LLMApi): The language model API engine to use.
        prompt (str, optional): The initial prompt for the agent. If left empty, a modular prompt is generated using the interests and behavior arguments. Defaults to an empty string.
        interests (list, optional): The interests of the agent. Defaults to an empty list.
        behavior (list, optional): The behavior of the agent. Defaults to an empty list.
        n_last_messages (int, optional): The number of messages that gets considered when generating memories. Defaults to 10.
        n_last_messages_reflection (int, optional): The number of messages that gets considered when generating reflections. Defaults to 20.    
        n_examples (int, optional): The number of examples to use in the agent's answer generation. Defaults to 3.
        reflect (bool, optional): Whether the agent should reflect on the last n_last_messages_reflection memories. Defaults to True.
        
        
        """
        super().__init__(name, llm, prompt, interests, behavior, n_last_messages, n_examples, react)
        self.n_last_messages_reflection = n_last_messages_reflection
        
    def get_reflections(self, n_memories):
        """
        Gets reflections from the last memories.
        
        Args:
        n_memories (int): The number of memories to use for reflections.
        
        """
        
        memories_obj=self.memory.search("SELECT id, text, turn, entry FROM txtai ORDER BY entry DESC", limit=n_memories)
        
        formatted_memories = "\n".join(x["text"] for x in memories_obj)
        
        reflection_prompt = f"""
You are trying to reflect on your memories, creating new short memories that build on the old ones.
You are reflecting on the last {n_memories} memories in your memory. Those are the memories:
###
{formatted_memories}
###
Give me 5 reflections on those memories, divided by a new line.
"""

        reflection_prompt_final = f"""
{self.prompt}\n
{reflection_prompt}
"""
        
        reflections_list = []
        # log the prompt for debugging
        self.log(reflection_prompt_final)
        
        while len(reflections_list) < 5:
            try:
                reflections = self.llm.generate_response(reflection_prompt_final)
                reflections_list = reflections.split("\n")
            except Exception as e:
                print(f"Reflections could not be generated: {e}. Retrying...")
                continue
            if len(reflections_list) < 5:
                print("Not enough reflections generated. Retrying...")
                continue
            reflections_list = [{"text": x, "turn": turn_count} for x in reflections_list]
        
        # save reflections in memory
        self.memory.upsert(reflections_list)
        
        return reflections_list
    
    def run_routines(self, turn_count, chat_history):
        """
        Runs routines for the agent. It will be called by the ChatThread object at the end of every turn.

        Args:
            turn_count (int): The current turn count.
            chat_history (list): The chat history.
        """
        if turn_count % self.n_last_messages == 0:
            print(f"Time to generate memories for {self.name}!")
            self.save_observations(self.get_observations(chat_history))
        if turn_count % self.n_last_messages_reflection == 0:
            print(f"Time to generate reflections for {self.name}!")
            self.get_reflections(self.n_last_messages)
        else:
            pass
            
            
            
def load_agent(agent_dict):
    """
    Loads an agent from a dictionary.
    
    Args:
    agent_dict (dict): The dictionary containing the agent data.
    
    """
    
    # construct an agent from the dictionary
    
    # first, the llm
    
    llm_class=agent_dict["llm"]
    llm = eval(llm_class)()
    
    # then the agent
    agent_type = agent_dict["type"]
    agent_name = agent_dict["name"]
    agent_prompt = agent_dict["prompt"]
    agent_interests = agent_dict["interests"]
    agent_behavior = agent_dict["behavior"]
    agent_n_examples = agent_dict["n_examples"]
    
    if agent_type == "Agent":
        agent = Agent(agent_name, llm, prompt=agent_prompt, interests=agent_interests, behavior=agent_behavior, n_examples=agent_n_examples)
    else:
        agent=eval(agent_type)(agent_name, llm, prompt=agent_prompt, interests=agent_interests, behavior=agent_behavior, n_examples=agent_n_examples)
        agent.memory.upsert(agent_dict["memories"])
    
    return agent


def load_chat(chat_path):
    """
    Loads a chat from a json.
    
    Args:
    chat_dict (dict): The dictionary containing the chat data.
    
    """
    
    with open(chat_path, "r") as f:
        chat_dict = json.load(f)
    
    # first, load the agents
    agent_list = [load_agent(x) for x in chat_dict["agent_list"]]
    
    # then, the chat thread
    chat_goal = chat_dict["chat_goal"]
    neutral_llm = eval(chat_dict["neutral_llm"])()
    sel_method =  chat_dict["sel_method"]
    n_eval = chat_dict["n_eval"]
    
    chat = ChatThread(agent_list=agent_list, chat_goal=chat_goal, neutral_llm=neutral_llm, sel_method=sel_method, n_eval=n_eval)
    chat.chat_history = chat_dict["chat_history"]
    chat.chat_evaluation = chat_dict["chat_evaluation"]
    chat.chat_id = chat_dict["chat_id"]
    
    return chat
