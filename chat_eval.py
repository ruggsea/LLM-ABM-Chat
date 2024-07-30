# implementing three evaluation metrics for dialogue quality evaluation: Perplexity, N-distinct and LLM-As-A-JUDGE

import os
import sys
import re
import numpy as np
import torch
import transformers
import json
import evaluate
import string
from llm_engines import ChatgptLLM, LLMApi
from dialogue_react_agent import load_base_prompt
from nltk.util import ngrams
import logging
# load gpt2 tokenizer
from transformers import GPT2Tokenizer
from tqdm import tqdm



# Set up logging
logging.basicConfig(filename='evaluation.log', level=logging.INFO, filemode="w", format="%(asctime)-15s %(message)s", force=True)


absolute_eval_prompt_path="prompts/absolute_eval.j2"

absolute_eval_prompt=load_base_prompt(absolute_eval_prompt_path)

pairwise_eval_prompt_path="prompts/pairwise_eval.j2"

pairwise_eval_prompt=load_base_prompt(pairwise_eval_prompt_path)

def load_chat_history(path:str):
    ## chat histories are json files with a list of chat messages, each message is an array of turn, speaker, text
    # it should be put in Bob: message\nAlice: message format
    
    # the path is a single line json, load it
    with open(path, "r") as f:
        chat_history = json.loads(f.readline())
    
    # now, every message is a list of turn, speaker, text, we want to convert it to a single string
    chat_str=""
    for message in chat_history:
        chat_str+=message[1].strip()+": "+message[2].strip()+"\n"
    return chat_str


def calc_perplexity(chat_history:str|list[str]):
    ## use hf evaluate to calculate per chat history perplexity
    ## returns a vector of perplexity scores (one per chat)
    # load the model
    
    if type(chat_history)==str:
        chat_history = [chat_history]
    # make only the second graphics card available
    
    # Split chat history into batches
    batch_size = 8
    chat_batches = [chat_history[i:i+batch_size] for i in range(0, len(chat_history), batch_size)]
    
    perplexities = []
    
    # Process each batch
    for batch in chat_batches:
        perplexity=evaluate.load("perplexity", module_type="metric")
        batch_results = perplexity.compute(model_id='gpt2',
                                           add_start_token=False,
                                           predictions=batch,
                                           max_length=1024,
                                           device="cuda")
            # clean gpu memory
        torch.cuda.empty_cache()
        del perplexity
        perplexities.extend(batch_results["perplexities"])
    
    
    
    return perplexities

def calc_distinct_n(chat_history:str|list[str], n:int=1):
    # calculate different n-grams in the chat history
    # return vector of n_distincts scores (one per chat)
    if type(chat_history)==str:
        chat_history = [chat_history]
    # for chat in chat_history, calculate the n-grams of size n
    n_distincts = []
    for chat in chat_history:
        # n of different n-grams
        # first substitute \n with space
        chat = chat.replace("\n", " ")
        # make everything lowercase
        chat = chat.lower()
        n_distinct = len(set(ngrams(chat.split(), n)))
        # make the n_distinct proportional to the length of the chat
        n_distinct = n_distinct/len(chat.split())
        n_distincts.append(n_distinct)
    return n_distincts


# now we need to calculate the LLM-As-A-JUDGE

def calc_llm_as_a_judge(chat_history:str|list[str], model:str="gpt-4o", n_consistency:int=1):
    # calculate the LLM-As-A-JUDGE metric
    # return vector of scores (one per chat)
    # format chat_history to a list of strings if it is a single string
    if type(chat_history)==str:
        chat_history = [chat_history]

    if model=="prometheus-2.0":
        # use the new model
        model="prometheus-7b-v2.0.Q5_0.gguf"
        llm=LLMApi()
        llm.model=model
    else:
        # set correct model
        llm=ChatgptLLM()
        llm.model=model        
        
    # the prompt is a jinja template
    prompt = absolute_eval_prompt
    
    scores = []
    
    def extract_score(llm_answer:str):
        # the score will be the number after the final [RESULT] token
        # it should be a number between 0 and 1     
        result_token="[RESULT]"
        # find the position of last result_token
        result_pos = llm_answer.rfind(result_token)
        if result_pos==-1:
            logging.info(f"No result token found in the answer: {llm_answer}")
            assert False, f"No result token found in the answer: {llm_answer}"
        # get the substring after the result_token
        llm_answer = llm_answer[result_pos+len(result_token):]
        # now, the score should be the first non white space char
        # find the first non white space char
        match = re.match(r"(\s*)(\S+)", llm_answer)
        if match:
            score = float(match.group(2))            
            if score<0 or score>5:
                logging.info(f"Invalid score: {score}")
                print(llm_answer)
                assert False, f"Invalid score: {score}"
            return score
        else:
            logging.info(f"Invalid answer: {llm_answer}")
            print(llm_answer)
            assert False, f"Invalid answer: {llm_answer}"
    logging.info(f"Judging singularly {len(chat_history)} chat histories with model {llm.model}")
    for chat in tqdm(chat_history, desc="Absolute evaluation"):
        single_chat_scores=[]
        prompt_rendered = prompt.render(chat_history=chat)
        logging.info(f"Prompt rendered: {prompt_rendered}")
        logging.info(f"Generating {n_consistency} scores for this chat history.")
        while len(single_chat_scores)<n_consistency:
            score = None
            while not score:
                try:
                    score = extract_score(llm.generate_response(prompt_rendered))
                except:
                    pass
            single_chat_scores.append(score)
            logging.info(f"Score generated: {score}")
        logging.info(f"Overall scores for this chat history: {np.mean(single_chat_scores)}")
        scores.append(np.mean(single_chat_scores))
        
    if len(chat_history)>1:
        logging.info(f"Mean score: {np.mean(scores)}")
    else:
        logging.info(f"Score: {scores[0]}")
    return scores


## same function but pairwise, for each pair of chat histories, calculate the LLM-As-A-JUDGE score

def calc_llm_as_a_judge_pairwise(chat_history_a:str|list[str], chat_history_b:str|list[str], model:str="gpt-4o", n_consistency:int=1):
    # calculate the LLM-As-A-JUDGE metric
    # output the winner of each pair of chat histories as a list
    
    prompt=pairwise_eval_prompt
    # format chat_history to a list of strings if it is a single string
    if type(chat_history_a)==str:
        chat_history_a = [chat_history_a]
    if type(chat_history_b)==str:
        chat_history_b = [chat_history_b]
        
    # check if the chat histories have the same length
    assert len(chat_history_a)==len(chat_history_b), "Chat histories must have the same length"
    
    if model=="prometheus-2.0":
        # use the new model
        model="prometheus-7b-v2.0.Q5_0.gguf"
        llm=LLMApi()
        llm.model=model
    else:
        # set correct model
        llm=ChatgptLLM()
        llm.model=model        
        
    def extract_score(llm_answer:str):
        """ Extract the winner of the chat history from the LLM answer (A or B)
        If the answer is not in the correct format, raise an exception
        Input: llm_answer: str
        Output: score: str (either A or B)
        """
        
        # the score will be the number after the final [RESULT] token
        # it should be either A or B
        result_token="[RESULT]"
        # find the position of last result_token
        result_pos = llm_answer.rfind(result_token)
        if result_pos==-1:
            logging.info(f"No result token found in the answer: {llm_answer}")
            assert False, f"No result token found in the answer: {llm_answer}"
        # get the substring after the result_token
        llm_answer = llm_answer[result_pos+len(result_token):]
        # now, the score should be the first non white space char
        # find the first non white space char
        match = re.match(r"(\s*)(\S+)", llm_answer)
        if match:
            score = match.group(2)            
            if score not in ["A", "B"]:
                logging.info(f"Invalid score: {score}")
                assert False, f"Invalid score: {score}"
            return score
        else:
            logging.info(f"Invalid answer: {llm_answer}")
            assert False, f"Invalid answer: {llm_answer}"
    logging.info(f"Judging pairwise {len(chat_history_a)} chat of group A with {len(chat_history_b)} of group b chat histories with model {llm.model}")
    matchup_winners=[]
    for chat_a, chat_b in tqdm(zip(chat_history_a, chat_history_b), total=len(chat_history_a), desc="Pairwise evaluation"):
        # shuffle the order of the chats
        single_comparison_winners=[]
        logging.info(f"Generating {n_consistency} winners for this pair of chat histories.")
        # use majority voting to get the final matchup winner 
        while len(single_comparison_winners)<n_consistency:
            # shuffling to stop favoring one chat history
            if np.random.rand()>0.5:
                chat_a, chat_b = chat_b, chat_a
                switch=True
            prompt_rendered = prompt.render(chat_history_a=chat_a, chat_history_b=chat_b)
            winner_this_round = None
            while not winner_this_round:
                try:
                    winner_this_round = extract_score(llm.generate_response(prompt_rendered))
                    # fix the winner if the chat histories were switched
                    if switch:
                        winner_this_round = "A" if winner_this_round=="B" else "B"
                except:
                    pass
            
            single_comparison_winners.append(winner_this_round)
            logging.info(f"Winner generated: {winner_this_round}")
        winner_this_matchup = max(set(single_comparison_winners), key=single_comparison_winners.count)
        # log match wins per chat history
        logging.info(f"Consistency level wins for A: {single_comparison_winners.count('A')}")
        logging.info(f"Consistency level wins for B: {single_comparison_winners.count('B')}")
        logging.info(f"Overall winner for this pair of chat histories: {winner_this_matchup}")
        matchup_winners.append(winner_this_matchup)
        
    # log the wins per groups of chat histories
    logging.info(f"Matchup wins for A: {matchup_winners.count('A')}")
    logging.info(f"Matchup wins for B: {matchup_winners.count('B')}")
    
    # return array of winners
    return matchup_winners
        
    
    
if __name__ == "__main__":
    chat_history = [
        "Alice: Hi,\nBob: Hello,\nAlice: How are you?\nBob: I'm fine, thank you.",
        "Alice: Hi,\nBob: Hello,\nAlice: How are you?\nBob: Not bad",
        
        
    ]
    
    judge_score = calc_llm_as_a_judge(chat_history, n_consistency=3, model="prometheus-2.0")
    print(f"LLM-As-A-JUDGE score: {judge_score}")
    
    # test pairwise
    chat_history_a = [
        "Alice: Hi,\nBob: Hello,\nAlice: How are you?\nBob: I'm fine, thank you.",
        "Alice: Hi,\nBob: Hello,\nAlice: How are you?\nBob: Not bad",
    ]
    chat_history_b = [
        "Alice: Hi,\nBob: Hello,\nAlice: How are you?\nBob: I hate you",
        "Alice: Hi,\nBob: Hello,\nAlice: The pen is on the table\nBob: Not bad",
    ]   
    
    pairwise_matches=calc_llm_as_a_judge_pairwise(chat_history_a, chat_history_b, n_consistency=3, model="prometheus-2.0")
    print(f"Pairwise winners: {pairwise_matches}")
