import os, json, random
from agent_factory import create_groupchat, unique_topics, get_persona_by_topics, gen_name
from dialogue_react_agent import DialogueReactAgent
from groupchat_thread import GroupChatThread
from llm_engines import LLMApi
from tqdm import tqdm
import sys

    

def load_chat_pairs(path:str):
    # the file is a json array of dictionaries, each dictionary has the keys "persona1", "persona2", "name1", "name2"
    chat_pairs=[]
    with open(path) as f:
        chat_pairs=json.load(f)
    return chat_pairs


def group_chats_from_chat_pairs(chat_pairs, **agent_args):
    groupchats=[]
    for chat_pair in chat_pairs:
        agent1=DialogueReactAgent(name=chat_pair["name1"], persona=chat_pair["persona1"], **agent_args)
        agent2=DialogueReactAgent(name=chat_pair["name2"], persona=chat_pair["persona2"], **agent_args)
        groupchat=GroupChatThread([agent1, agent2], neutral_llm=agent_args["llm"], n_eval=-1)
        groupchats.append(groupchat)
    return groupchats

def main():
    # make print compatible with tqdm
    tqdm.monitor_interval = 0
    
    
    # load the chat pairs
    chat_pairs=load_chat_pairs("thesis_generations/personas_pairs.json")
    llm=LLMApi(model="turboderp_gemma-2-27b-it-exl2_5.0bpw")
    # chat_ids for each ablation
    chat_ids_no_ablation=[]
    i=0
    
    
    
    if not os.path.exists("thesis_generations/gen_chat_ids_list_gemma2.json"):
        os.makedirs("thesis_generations", exist_ok=True)
        chats=group_chats_from_chat_pairs(chat_pairs, llm=llm,memory_freq=10, reflections_freq=25, ablation=None)
        # to be sure, print the llm of the first chat
        print(f"Agents using model: {chats[0].agent_list[0].llm.model}")

        for chat in tqdm(chats, desc="Running chats (no ablation)"):
            ## add offset to chat id
            i+=1
            chat.chat_id+=str(i)
            id=chat.chat_id           
            chat.run_chat()
            chat_ids_no_ablation.append(id)
            # save the chat ids
            with open("thesis_generations/gen_chat_ids_list_gemma2.json", "w") as f:
                json.dump(chat_ids_no_ablation, f)
    
    # if it exists, load the chat ids and continue
    else:
        with open("thesis_generations/gen_chat_ids_list_gemma2.json") as f:
            chat_ids_no_ablation=json.load(f)
        if len(chat_pairs)> len(chat_ids_no_ablation):
            starting_index=len(chat_ids_no_ablation)
            print("Resuming generation from last chat id")
            print(f"Chat ids generated: {len(chat_ids_no_ablation)}")
            print(f"Chat pairs to generate: {len(chat_pairs)}")
            print(f"Chat pairs left to generate: {len(chat_pairs[starting_index:])}")
            chats=group_chats_from_chat_pairs(chat_pairs[starting_index:], llm=llm,memory_freq=10, reflections_freq=25, ablation=None)
            print(f"Generating {len(chats)} chats")
            for chat in tqdm(chats, desc="Running chats (no ablation)"):
                ## add offset to chat id
                i+=1
                chat.chat_id+=str(i)    
                id=chat.chat_id           
                chat.run_chat()
                chat_ids_no_ablation.append(id)
                # save the chat ids
                with open("thesis_generations/gen_chat_ids_list_gemma2.json", "w") as f:
                    json.dump(chat_ids_no_ablation, f)
            

    # ablation 3: no_dialogue_no_react
    
    chat_ids_no_dialogue_no_react=[]
    
    if not os.path.exists("thesis_generations/gen_chat_ids_list_no_dialogue_no_react_gemma2.json"):
        chats=group_chats_from_chat_pairs(chat_pairs, llm=llm,memory_freq=10, reflections_freq=25, ablation="no_dialogue_no_react")
        i=0
        for chat in tqdm(chats, desc="Running chats (no dialogue, no react)"):
            ## add offset to chat id
            i+=1
            chat.chat_id+=str(i)
            id=chat.chat_id
            chat.run_chat(max_turns=50)
            chat_ids_no_dialogue_no_react.append(id)
            # save the chat ids
            with open("thesis_generations/gen_chat_ids_list_no_dialogue_no_react_gemma2.json", "w") as f:
                json.dump(chat_ids_no_dialogue_no_react, f)
            
    # ablation 4: no_dialogue
    
    chat_ids_no_dialogue=[]
    
    if not os.path.exists("thesis_generations/gen_chat_ids_list_no_dialogue_no_react_gemma2.json"):
        chats=group_chats_from_chat_pairs(chat_pairs, llm=llm,memory_freq=10, reflections_freq=25, ablation="no_dialogue")
        i=0
        for chat in tqdm(chats, desc="Running chats (no dialogue)"):
            ## add offset to chat id
            i+=1
            chat.chat_id+=str(i)
            id=chat.chat_id           
            chat.run_chat(max_turns=50)
            chat_ids_no_dialogue.append(id)
            # save the chat ids
            with open("thesis_generations/gen_chat_ids_list_no_dialogue_no_react_gemma2.json", "w") as f:
                json.dump(chat_ids_no_dialogue, f)
                
if __name__=="__main__":
    main()