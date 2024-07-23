# the chats ids are in the files gen_chat_ids_list.json, gen_chat_ids_list_no_reflections.json, gen_chat_ids_list_no_memory.json, gen_chat_ids_list_no_dialogue_no_react.json, gen_chat_ids_list_no_dialogue_no_react.json and naive_gen_chat_ids_list.json

# to run the eval, first load the chat history from the chat ids, then run the eval
import json
import os
import logging
from tqdm import tqdm
from chat_eval import calc_perplexity, calc_distinct_n, calc_llm_as_a_judge_pairwise, calc_llm_as_a_judge
import random

# setup logging for the evaluation
logging.basicConfig(filename="running_eval.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Create a file handler and set its level to INFO
file_handler = logging.FileHandler("running_eval.log")
file_handler.setLevel(logging.INFO)

# Create a formatter and set it to the file handler
formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger = logging.getLogger()
logger.addHandler(file_handler)

def load_chat_history(path, max_messages=1000):
    with open(path) as f:
        chat_messages = json.load(f)
    # format from id, speaker, message to speaker: message
    chat_history = ""
    for msg in chat_messages[:max_messages]:
        chat_history += msg[1] + ": " + msg[2] + "\n"
    
    return chat_history

def load_list_of_chat_ids(list_path):
    with open(list_path) as f:
        chat_ids = json.load(f)
    return chat_ids

def load_chats_from_chat_ids(chat_ids, chat_dir, chat_prefix="", max_messages=1000):
    chats=[]
    for chat_id in chat_ids:
        chat_path = os.path.join(chat_dir, chat_prefix + chat_id + ".json")
        chat=load_chat_history(chat_path, max_messages=max_messages)
        chats.append(chat)
    return chats

def plot_eval_results(eval_results, title, x_label, y_label, save_path):
    import matplotlib.pyplot as plt
    # plot results as a box plot, always using the labels naive - no_reflections - no_memory - no_dialogue_no_react - no_dialogue - dialogue_react

    
    labels = ["naive", "no_dialogue_no_react", "no_dialogue", "no_memory", "no_reflections", "dialogue_react"]
    x = range(len(labels))
    x=[i+1 for i in x]
    
    # add width to the boxplot
    plt.figure(figsize=(10,6))
    
    # make sorted list from eval results dict
    
    sorted_results = [eval_results[label] for label in labels]
    
    plt.boxplot(sorted_results)
    
    plt.xticks(x, labels)
    
    plt.title(title)
    
    plt.xlabel(x_label)
    
    # labels should be Naive, No dialogue no react, No dialogue, No memory, No reflections, Dialogue React
    
    plt.xticks(x, ["Naive", "No dialogue\nno react", "No dialogue", "No memory", "No reflections", "Dialogue React"])

    
    # tilt the x labels but keep them aligned
    #plt.xticks(rotation=45)
    
    plt.ylabel(y_label)
    
    plt.savefig(save_path, bbox_inches="tight")
    
    plt.clf()
    

if __name__=="__main__":
    
    # create eval.json or load it if it exists
    if os.path.exists("thesis_generations/eval.json"):
        with open("thesis_generations/eval.json") as f:
            eval_dict = json.load(f)
    else:
        eval_dict = {}    
    
    # load naive chat ids
    naive_chat_ids = load_list_of_chat_ids("thesis_generations/naive_gen_chat_ids_list.json")
    # load chat ids for the ablations
    chat_ids_no_reflections = load_list_of_chat_ids("thesis_generations/gen_chat_ids_list_no_reflections.json")
    chat_ids_no_memory = load_list_of_chat_ids("thesis_generations/gen_chat_ids_list_no_memory.json")
    chat_ids_no_dialogue_no_react = load_list_of_chat_ids("thesis_generations/gen_chat_ids_list_no_dialogue_no_react.json")
    chat_ids_no_dialogue = load_list_of_chat_ids("thesis_generations/gen_chat_ids_list_no_dialogue.json")
    # now the non ablation chat ids
    chat_ids_dialogue_react = load_list_of_chat_ids("thesis_generations/gen_chat_ids_list.json")
    
    # load the chats
    chat_dir = "chat_history/"
    chat_prefix = "react_chat_history_"
    naive_prefix = "naive_chat_history_"
    
    naive_chats = load_chats_from_chat_ids(naive_chat_ids, chat_dir, chat_prefix=naive_prefix)
    logging.info(f"Loaded {len(naive_chats)} naive chats")
    chat_no_reflections = load_chats_from_chat_ids(chat_ids_no_reflections, chat_dir, chat_prefix=chat_prefix)
    logging.info(f"Loaded {len(chat_no_reflections)} chats with no reflections")

    chat_no_memory = load_chats_from_chat_ids(chat_ids_no_memory, chat_dir, chat_prefix=chat_prefix)
    logging.info(f"Loaded {len(chat_no_memory)} chats with no memory")

    chat_no_dialogue_no_react = load_chats_from_chat_ids(chat_ids_no_dialogue_no_react, chat_dir, chat_prefix=chat_prefix)
    logging.info(f"Loaded {len(chat_no_dialogue_no_react)} chats with no dialogue and no reaction")

    chat_no_dialogue = load_chats_from_chat_ids(chat_ids_no_dialogue, chat_dir, chat_prefix=chat_prefix)
    logging.info(f"Loaded {len(chat_no_dialogue)} chats with no dialogue")

    chat_dialogue_react = load_chats_from_chat_ids(chat_ids_dialogue_react, chat_dir, chat_prefix=chat_prefix)
    logging.info(f"Loaded {len(chat_dialogue_react)} chats with dialogue and reaction")

    # same but loading only the first 10 messages per chat
    
    naive_chats_10 = load_chats_from_chat_ids(naive_chat_ids, chat_dir, chat_prefix=naive_prefix, max_messages=10)
    logging.info(f"Loaded {len(naive_chats_10)} naive chats with 10 messages")
    chat_no_reflections_10 = load_chats_from_chat_ids(chat_ids_no_reflections, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    logging.info(f"Loaded {len(chat_no_reflections_10)} chats with no reflections with 10 messages")
    chat_no_memory_10 = load_chats_from_chat_ids(chat_ids_no_memory, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    logging.info(f"Loaded {len(chat_no_memory_10)} chats with no memory with 10 messages")
    chat_no_dialogue_no_react_10 = load_chats_from_chat_ids(chat_ids_no_dialogue_no_react, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    logging.info(f"Loaded {len(chat_no_dialogue_no_react_10)} chats with no dialogue and no reaction with 10 messages")
    chat_no_dialogue_10 = load_chats_from_chat_ids(chat_ids_no_dialogue, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    chat_dialogue_react_10 = load_chats_from_chat_ids(chat_ids_dialogue_react, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    
    # make dict with the chats
    chat_dict = {"naive": naive_chats, "no_reflections": chat_no_reflections, "no_memory": chat_no_memory, "no_dialogue_no_react": chat_no_dialogue_no_react, "no_dialogue": chat_no_dialogue, "dialogue_react": chat_dialogue_react}
    chat_dict_10 = {"naive": naive_chats_10, "no_reflections": chat_no_reflections_10, "no_memory": chat_no_memory_10, "no_dialogue_no_react": chat_no_dialogue_no_react_10, "no_dialogue": chat_no_dialogue_10, "dialogue_react": chat_dialogue_react_10}
    
    
    # run distinct n for different ns on the chats, plot the results
    
    distinct_n_results = {}
    for key, value in chat_dict.items():
        for n in range(1,5):
            if n not in distinct_n_results:
                distinct_n_results[n] = {}
            distinct_n_results[n][key] = calc_distinct_n(value, n)
    
    
    # make sure thesis_generations/figures exists
    if not os.path.exists("thesis_generations/figures"):
        os.makedirs("thesis_generations/figures")
    
    # plot the results 
    for n, results in distinct_n_results.items():
        plot_eval_results(results, f"Distinct {n}-grams for different approaches", "Approach", f"Distinct {n}-grams", f"thesis_generations/figures/distinct_{n}_grams.png")
        
    # same but first 10 messages
    distinct_n_results_10 = {}
    
    for key, value in chat_dict_10.items():
        for n in range(1,5):
            if n not in distinct_n_results_10:
                distinct_n_results_10[n] = {}
            distinct_n_results_10[n][key] = calc_distinct_n(value, n)
        
    # plot the results
    for n, results in distinct_n_results_10.items():
        plot_eval_results(results, f"Distinct {n}-grams for different approaches (first 10 messages)", "Approach", f"Distinct {n}-grams", f"thesis_generations/figures/distinct_{n}_grams_10.png")
        
    # running distinct n for the whole generation (cat of chat histories with the same approach)
    # load the chat histories
    
    complete_naive_chat = "\n".join(naive_chats)
    complete_no_reflections_chat = "\n".join(chat_no_reflections)
    complete_no_memory_chat = "\n".join(chat_no_memory)
    complete_no_dialogue_no_react_chat = "\n".join(chat_no_dialogue_no_react)
    complete_no_dialogue_chat = "\n".join(chat_no_dialogue)
    complete_dialogue_react_chat = "\n".join(chat_dialogue_react)
    
    complete_chats = {"naive": complete_naive_chat, "no_reflections": complete_no_reflections_chat, "no_memory": complete_no_memory_chat, "no_dialogue_no_react": complete_no_dialogue_no_react_chat, "no_dialogue": complete_no_dialogue_chat, "dialogue_react": complete_dialogue_react_chat}
    
    complete_chats_10 = {}
    for key, value in chat_dict_10.items():
        complete_chats_10[key] = "\n".join(value)
    
    # run distinct n for the complete chats
    
    complete_distinct_n_results = {}
    for key, value in complete_chats.items():
        for n in range(1,5):
            if n not in complete_distinct_n_results:
                complete_distinct_n_results[n] = {}
            complete_distinct_n_results[n][key] = calc_distinct_n(value, n)
    
    # plot the results
    for n, results in complete_distinct_n_results.items():
        plot_eval_results(results, f"Distinct {n}-grams for different approaches (complete chats)", "Approach", f"Distinct {n}-grams", f"thesis_generations/figures/distinct_{n}_grams_complete.png")
        
    # same but for the first 10 messages
    
    complete_distinct_n_results_10 = {}
    
    for key, value in complete_chats_10.items():
        for n in range(1,5):
            if n not in complete_distinct_n_results_10:
                complete_distinct_n_results_10[n] = {}
            complete_distinct_n_results_10[n][key] = calc_distinct_n(value, n)
            
    # plot the results
    
    for n, results in complete_distinct_n_results_10.items():
        plot_eval_results(results, f"Distinct {n}-grams for different approaches (first 10 messages)", "Approach", f"Distinct {n}-grams", f"thesis_generations/figures/distinct_{n}_grams_complete_10.png")
    
    eval_dict["distinct_n"] = distinct_n_results
    eval_dict["distinct_n_10"] = distinct_n_results_10
    eval_dict["complete_distinct_n"] = complete_distinct_n_results
    ### Perplexity
    
    # run perplexity for the chats
    
    if "perplexity" not in eval_dict:
        perplexity_results = {}
        for key, value in chat_dict.items():
            perplexity_results[key] = calc_perplexity(value)
        eval_dict["perplexity"] = perplexity_results

    # plot the results
    plot_eval_results(eval_dict["perplexity"], "Perplexity for different approaches", "Approach", "Perplexity", "thesis_generations/figures/perplexity.png")
    
    # same but for the first 10 messages
    
    if "perplexity_10" not in eval_dict:
        perplexity_results_10 = {}
        for key, value in chat_dict_10.items():
            perplexity_results_10[key] = calc_perplexity(value)
        eval_dict["perplexity_10"] = perplexity_results_10
    # plot the results
    plot_eval_results(eval_dict["perplexity_10"], "Perplexity for different approaches (first 10 messages)", "Approach", "Perplexity", "thesis_generations/figures/perplexity_10.png")
    
    # perplexity but normalized by the number of words
    if "perplexity_normalized" not in eval_dict:
        perplexity_results_normalized = {}
        
        for key, value in chat_dict.items():
            num_words_vector = [len(chat.split()) for chat in value]
            perplexity_results_normalized[key] = [perplexity/num_words for perplexity, num_words in zip(calc_perplexity(value), num_words_vector)]
        eval_dict["perplexity_normalized"] = perplexity_results_normalized
            
    # plot the results
    
    plot_eval_results(eval_dict["perplexity_normalized"], "Perplexity normalized by the number of words for different approaches", "Approach", "Perplexity", "thesis_generations/figures/perplexity_normalized.png")
        
    # dump the eval_dict to a json file
    with open("thesis_generations/eval.json", "w") as f:
        json.dump(eval_dict, f)
    
    ## LLM as a judge ##

    # Check if llm_as_judge results are already present in eval_dict
    if "llm_as_judge" not in eval_dict:
        # run llm as a judge for the chats for the first 10 messages
        llm_as_judge_results = {}
        for key, value in chat_dict_10.items():
            llm_as_judge_results[key] = calc_llm_as_a_judge(value,model="prometheus-2.0", n_consistency=3)
            
        # plot the results
        plot_eval_results(llm_as_judge_results, "GPT4o as a judge for different approaches (first 10 messages)", "Approach", "LLM as a judge", "thesis_generations/figures/llm_as_judge.png")
        
        eval_dict["llm_as_judge"] = llm_as_judge_results
        
        # dump the eval_dict to a json file
        with open("thesis_generations/eval.json", "w") as f:
            json.dump(eval_dict, f)

    # Check if llm_as_judge_pairwise results are already present in eval_dict
    if "llm_as_judge_pairwise" not in eval_dict:
        # same but pairwise
        llm_as_judge_pairwise_results = {}
        
        # do matchups between naive, no_dialogue_no_react, dialogue_react
        matches = [("naive", "no_dialogue_no_react"), ("naive", "dialogue_react"), ("no_dialogue_no_react", "dialogue_react")]
        
        for match in matches:
            key = match[0] + "_" + match[1]
            llm_as_judge_pairwise_results[key] = calc_llm_as_a_judge_pairwise(chat_dict_10[match[0]], chat_dict_10[match[1]], model="prometheus-2.0", n_consistency=3)    
        
        eval_dict["llm_as_judge_pairwise"] = llm_as_judge_pairwise_results
        with open("thesis_generations/eval.json", "w") as f:
            json.dump(eval_dict, f)
    # a is the first competitor, b is the second competitor
    # plot a wins and b wins barplots for each matchup
    # a is the first competitor, b is the second competitor
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10,6))
    
    for key, value in llm_as_judge_pairwise_results.items():
        a_wins = value.count("A")
        b_wins = value.count("B")
        x = np.arange(2)
        plt.bar(x, [a_wins, b_wins])
        competitors = key.split("_")
        plt.xticks(x, competitors)
        plt.title(f"{key} wins")
        plt.savefig(f"thesis_generations/figures/{key}_wins.png", bbox_inches="tight")
        plt.clf()
        
        
    