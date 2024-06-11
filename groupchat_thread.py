from llm_engines import LLMApi, ChatgptLLM
import json, random, time, textwrap, logging, os, re
from dialogue_react_agent import DialogueReactAgent, load_base_prompt

eval_prompt=load_base_prompt("prompts/chat_evaluation.j2")

logging.basicConfig(level=logging.INFO, filename="chat.log", filemode="w", format="%(asctime)-15s %(message)s")

class GroupchatThread:
    def __init__(self, agent_list=[], neutral_llm=LLMApi(), sel_method="random", n_eval=10):
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
        self.neutral_llm = neutral_llm
        self.sel_method = sel_method
        self.n_eval = n_eval
        self.eval_prompt = eval_prompt
        # conversation turn, 1 for first message
        self.turn = 0
        
        # storing chat evals
        self.chat_evaluation = {}
        
        # unique identifier for the chat
        self.chat_id = f"chat_{str(int(time.time()))}"
        
        
        # conversation starters
        self.conversation_starters = ["Hi!", "Hello!", "How are you?", "How is it going?", "What's up?", "How are you doing?"]


        # agent colors: each agent is assigned a different ansi color - a dict of agent_name: ansi_color
        self.agent_colors = {}
        for i, agent in enumerate(self.agent_list):
            agent_color = f"\033[9{i+2}m"
            agent.color = agent_color
            self.agent_colors[agent.name] = agent_color 
            
        
    
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
        Starts the conversation with a random agent.
        """
        # should only work if turn is 0
        assert self.turn == 0, "The conversation has already started."
        
        # log the start of the conversation and agent list
        logging.info(f"Starting conversation {self.chat_id} with agents: {[agent.name for agent in self.agent_list]}")
        
        random_agent = self.pick_random_agent()
        first_message = (1, random_agent.name, random.choice(self.conversation_starters))
        # after the first message, the turn is set to 1
        self.turn += 1
        self.chat_history.append(first_message)
        
        return first_message
    
    def get_chat_answer(self, last_messages, agent):
        other_agents= [a for a in self.agent_list if a != agent]
        other_agents_names = [a.name for a in other_agents]
        answer= agent.get_answer(last_messages, agent_list=other_agents_names, n_agents=len(self.agent_list), turn_count=self.turn)
        # increase turn count
        self.turn += 1
        # append to chat history
        self.chat_history.append((self.turn, agent.name, answer))
        
        return answer
        
    def render_last_message(self):
        """
        Renders the last message in the chat history.
        """
        # check if there is a last message
        assert len(self.chat_history) > 0, "There are no messages in the chat history."
        
        last_message = self.chat_history[-1]
        n_message, agent, message =  last_message[0], last_message[1], last_message[2]
        
        # color agent name based on agent index
        agent_color = self.agent_colors[agent]
        # print message
        msg_string = f"{agent_color}{agent}\033[0m: {message}"
        
        wrapped_message = textwrap.fill(msg_string, width=80, subsequent_indent=' ' * 4)
        print(wrapped_message)
    
    def evaluate_chat(self, start_index=0, end_index=-1):
        """
        Evaluates the chat based on the last n_eval turns.
        
        Parameters:
        start_index (int, optional): The start index for the evaluation. Defaults to 0.
        end_index (int, optional): The end index for the evaluation. Defaults to -1.
        
        Returns:
            dict: The chat evaluation.
        """
        
        messages_to_evaluate = self.chat_history[start_index:end_index]
        
        # messages in agentname: message format
        messages_eval_string= [f"{agent}: {message}" for _, agent, message in messages_to_evaluate]
        
        eval_prompt_filled=self.eval_prompt.render(messages="\n".join(messages_eval_string))
        
        ## generate evaluation 3 times and take the average, each time make sure the answer is valid
        eval_results=[]
        
        while len(eval_results) < 3:
            try:
                eval_result = self.neutral_llm.generate_response(eval_prompt_filled)
                # try finding number/5## in the response, it should be the number before /5 followed by ##
                pattern = r"\d+\s*/\s*5\s*##"
                eval_score = re.search(pattern, eval_result).group(0)
                # get the number before /5
                eval_score = eval_score.split("/")[0]                
                eval_score = float(eval_score)
                if eval_score >= 0 and eval_score <= 10:
                    eval_results.append(eval_score)
            except Exception as e:
                logging.error(f"Error in evaluation response: {eval_result}")
                print(e)
        
        eval_score = sum(eval_results) / 3
        
        self.chat_evaluation[self.turn] = eval_score
        print(f"Chat evaluation: {eval_score}")
        return eval_score
        
            
        
    def dump_chat(self):
        """
        Dumps the chat history to a JSON file.
        """
        chat_data = {"chat_id": self.chat_id, "chat_history": self.chat_history, "chat_evaluation": self.chat_evaluation, "agent_list": [agent.dump_agent() for agent in self.agent_list], "neutral_llm": self.neutral_llm.__class__.__name__, "sel_method": self.sel_method, "n_eval": self.n_eval}
        if not os.path.exists("chat_logs"):
            os.makedirs("chat_logs")
        with open(f"chat_logs/{self.chat_id}.json", "w") as f:
            json.dump(chat_data, f)    
            
        
    
    
    def run_chat(self, max_turns=50):
        """
        Runs the chat for a maximum number of turns. As of now, the simulation uses a random agent selection method.
        
        Args:
            max_turns (int, optional): The maximum number of turns for the chat. Defaults to 50.
        
        Returns:
            list: The chat history.
        """
        
        self.start_conversation()
        self.render_last_message()

        # at this point, turn is 1
        
        # main chat loop runs until max_turns is reached
        while self.turn < max_turns:
            
            # run agent routines
            for agent in self.agent_list:
                agent.run_routines(self.turn, self.chat_history, self.agent_list, len(self.agent_list))
            
            # select agent to answer
            if self.sel_method == "random":
                # select random agent
                random_agent = self.pick_random_agent()
                # make sure that the agent is not the last one to send a message
                while self.chat_history[-1][1] == random_agent.name:
                    random_agent = self.pick_random_agent()
                
                # get chat history (limited to the last 10 messages)
                last_messages = self.chat_history[-10:]
            
                # get answer from agent
                answer = self.get_chat_answer(last_messages, random_agent)
                # render last message
                self.render_last_message()
                
                # evaluate chat
                if self.n_eval != -1 and self.turn % self.n_eval == 0:
                    self.evaluate_chat()
                    
                
            else:
                raise ValueError("The selection method is not valid.")
        
        # log the end of the conversation
        logging.info(f"Ending conversation {self.chat_id} after {self.turn} turns.")
        # at the end of the chat, evaluate the chat
        if self.n_eval != -1:
            self.evaluate_chat()
        
        self.dump_chat()
        
        return self.chat_history
        
        
    
    