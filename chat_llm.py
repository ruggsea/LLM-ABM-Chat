
from llm_engines import LLMApi
import random, textwrap, logging
from txtai import Embeddings


logging.basicConfig(level=logging.INFO, filename="chat.log", filemode="w", format="%(asctime)-15s %(message)s")


class Agent:
    def __init__(self, name, llm, prompt="", interests=[], behavior=[]): 
        """
        Initializes an Agent object.

        Args:
            name (str): The name of the agent.
            llm (LLMApi): The LLM engine, it defaults to LLMAPI.
            prompt (str, optional): The initial prompt for the agent. If left empty, a modular prompt is generated using the interests and behavior arguments. Defaults to an empty string.
            interests (list, optional): The interests of the agent. Defaults to an empty list.
            behavior (list, optional): The behavior of the agent. Defaults to an empty list.
        """
        self.name = name
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
        
        seek_answer_prompt = f"""
        Send a message to the group. Keep your answers short and casual and do not sound too excited but try to keep the conversationg going as much as possible. Be creative and engaging.
        Provide your answer as you would text it. Avoid greeting people and saying hi if you have already greeted them before.
        The last messages (they are shown in the format "message_number. name: message") were:
        ###
        {formatted_last_messages_str}
        ###
        
        How do you answer? Give your answer as you would text it, in the format "message". End your answer with ##.  Some examples:
        
        "Hey, everything is fine. I am just chilling. ##"
        "I am doing great. ##"
        "I like that one ##"
        "message ##"
        
        Your answer:  
        ###
        {self.name}:"""
        
        if extra_context != "":
            seek_answer_prompt = f"{self.prompt}{extra_context}\n{seek_answer_prompt}"
        else:
            seek_answer_prompt = f"{self.prompt}\n{seek_answer_prompt}"
        
        agent_answer = ""
        
        self.log(seek_answer_prompt)
        
        while len(agent_answer) < 5:
            agent_answer = self.llm.generate_response(seek_answer_prompt)
            if agent_answer == "":
                print("Empty answer returned. Retrying...")
        
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


class ChatThread:
    def __init__(self, agent_list=[], chat_goal="There is not a specific goal for this conversation.", neutral_llm=LLMApi(), sel_method="random"):
        """
        Initializes a ChatThread object.

        Args:
            agent_list (list, optional): The list of agents participating in the chat. Defaults to an empty list.
            chat_goal (str, optional): The goal of the chat. Defaults to "There is not a specific goal for this conversation.".
            neutral_llm (LLMApi, optional): The language model API for LLM generation needs outside the scope of any agent. Defaults to LLMApi().
            sel_method (str, optional): The method for selecting the next agent to answer. Defaults to "random".
        """
        self.chat_history = []
        self.agent_list = agent_list
        self.chat_goal = chat_goal
            
        self.conversation_starters = ["Hi!", "Hello!", "How are you?", "How is it going?", "What's up?", "How are you doing?"]
        self.neutral_llm = neutral_llm
        self.sel_method = sel_method
        
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
        {self.chat_goal}. In this chat, there's you, {agent.get_name()}, and {other_agents_str}.
        """
        # check if the answer fits the validation criteria
        
        validation_ending="##"
        agent_answer = ""
        while not agent_answer.endswith(validation_ending):
            agent_answer = agent.get_answer(last_messages)
             # trim the answer to remove everything after the validation ending
            agent_answer = agent_answer[:agent_answer.find(validation_ending)]

            if agent_answer=="":
                print(f"Invalid answer. Retrying...")
                continue
            else:
                # add the validation ending
                agent_answer = agent_answer + validation_ending
            
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
        max_turns = max_turns + len(self.chat_history)

        starting_agent_name = self.chat_history[0][1]
        starting_agent = [agent for agent in self.agent_list if agent.get_name() == starting_agent_name][0]
        prev_agent = starting_agent
        
        global turn_count
        self.turn_count = len(self.chat_history)
        
        
        
        while self.turn_count <= max_turns:
            for agent in self.agent_list:
                agent.run_routines(self.turn_count, self.chat_history)
            
            if self.sel_method == "random":
                random_agent = self.pick_random_agent()
                while random_agent == prev_agent:
                    random_agent = self.pick_random_agent()

            last_messages = self.chat_history
            # update turn count and get answer for the turn
            self.turn_count += 1
            self.get_chat_answer(last_messages, random_agent)
            self.render_last_message()
            prev_agent = random_agent
            
            
        return self.chat_history
    

class MemoryAgent(Agent):
    def __init__(self, name, llm, prompt="", interests=[], behavior=[], n_last_messages=10):
        """
        Initializes a Agent object with memory.

        Args:
            name (str): The name of the agent.
            llm (LLMApi): The language model API.
            prompt (str, optional): The initial prompt for the agent. Defaults to an empty string.
            interests (list, optional): The interests of the agent. Defaults to an empty list.
            behavior (list, optional): The behavior of the agent. Defaults to an empty list.
            n_last_messages (int, optional): The number of messages that gets considered when generating memories. Defaults to 10.
        """
        super().__init__(name, llm, prompt, interests, behavior)
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
            if len(observations_list) < 5:
                print("Not enough observations generated. Retrying...")

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
            formatted_last_messages_str.append(f"{message_n}. {agent}: {message}")
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        
        seek_answer_prompt = f"""
        Send a message to the group. Keep your answers short and casual and do not sound too excited but try to keep the conversationg going as much as possible. Be creative and engaging.
        Provide your answer as you would text it. Avoid greeting people and saying hi if you have already greeted them before.
        The last messages in the chat were (they are shown in the format "message_number. name: message"): 
        ###
        {formatted_last_messages_str}
        ###
        How do you answer? Give your answer as you would text it, in the format "message". End your answer with ##.  Some examples:
        
        "Hey, everything is fine. I am just chilling. ##"
        "I am doing great. ##"
        "I like that one ##"
        "message ##"
        
        Your answer:          
        ###
        {self.name}:
        """
        
        if self.memory.isdense():   
            last_message = last_messages[-1]
            memories = self.memory.search(last_message[-1], limit=5)
            context = "\n".join(x["text"] for x in memories)
            if extra_context != "":
                seek_answer_prompt_with_context = f"""
                {self.prompt}\n
                {extra_context}\n
                Send a message to the group. Keep your answers short and casual and do not sound too excited but try to keep the conversationg going as much as possible. Be creative and engaging.
                Provide your answer as you would text it. Avoid greeting people and saying hi if you have already greeted them before.
                The last messages in the chat were (they are shown in the format "message_number. name: message"): 
                ###
                {formatted_last_messages_str}
                ###
                How do you answer? Give your answer as you would text it, in the format "message". End your answer with ##.  Some examples:
                
                "Hey, everything is fine. I am just chilling. ##"
                "I am doing great. ##"
                "I like that one ##"
                "message ##"
                
                In addition, you remember the following memories relevant to chat right now:
                {context}\n
                
                Your answer:          
                ###
                {self.name}:
                """
            else:
                seek_answer_prompt_with_context = f"""
                {self.prompt}\n
                Send a message to the group. Keep your answers short and casual and do not sound too excited but try to keep the conversationg going as much as possible. Be creative and engaging.
                Provide your answer as you would text it. Avoid greeting people and saying hi if you have already greeted them before.
                The last messages in the chat were (they are shown in the format "message_number. name: message"): 
                ###
                {formatted_last_messages_str}
                ###
                How do you answer? Give your answer as you would text it, in the format "message". End your answer with ##.  Some examples:
                
                "Hey, everything is fine. I am just chilling. ##"
                "I am doing great. ##"
                "I like that one ##"
                "message ##"
                                
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
    
    


class ReflectingAgent(MemoryAgent):
    def __init__(self, name, llm, prompt="", interests=[], behavior=[], n_last_messages=10, n_last_messages_reflection=20):
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
        
        
        """
        super().__init__(name, llm, prompt, interests, behavior, n_last_messages)
        self.n_last_messages_reflection = n_last_messages_reflection
        
    def get_reflections(self, n_memories):
        """
        Gets reflections from the last memories.
        
        Args:
        n_memories (int): The number of memories to use for reflections.
        
        """
        
        memories_obj=self.memory.search("SELECT id, text, entry FROM txtai ORDER BY entry DESC", limit=n_memories)
        
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
            if len(reflections_list) < 5:
                print("Not enough reflections generated. Retrying...")
        
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
            
        