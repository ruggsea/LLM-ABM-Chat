# the chats ids are in the files gen_chat_ids_list.json, gen_chat_ids_list_no_reflections.json, gen_chat_ids_list_no_memory.json, gen_chat_ids_list_no_dialogue_no_react.json, gen_chat_ids_list_no_dialogue_no_react.json and naive_gen_chat_ids_list.json

# to run the eval, first load the chat history from the chat ids, then run the eval
import json
import os
import logging
from tqdm import tqdm
from chat_eval import calc_perplexity, calc_distinct_n
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

if __name__=="__main__":
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

    # show some examples
    # logging.info("Example naive chat:")
    # logging.info(naive_chats[0])
    # logging.info("Example chat with no reflections:")
    # logging.info(chat_no_reflections[0])
    # logging.info("Example chat with no memory:")
    # logging.info(chat_no_memory[0])
    # logging.info("Example chat with no dialogue and no reaction:")
    # logging.info(chat_no_dialogue_no_react[0])
    # logging.info("Example chat with no dialogue:")
    # logging.info(chat_no_dialogue[0])
    # logging.info("Example chat with dialogue and reaction:")
    # logging.info(chat_dialogue_react[0])


    # distinct 2 on complete chats
    logging.info("Calculating distinct 2 on complete chats")
    distinct_2_naive = calc_distinct_n(naive_chats, 2)
    logging.info(f"Distinct 2 for naive chat: {distinct_2_naive}")
    distinct_2_no_reflections = calc_distinct_n(chat_no_reflections, 2)
    logging.info(f"Distinct 2 for chat with no reflections: {distinct_2_no_reflections}")
    distinct_2_no_memory = calc_distinct_n(chat_no_memory, 2)
    logging.info(f"Distinct 2 for chat with no memory: {distinct_2_no_memory}")
    distinct_2_no_dialogue_no_react = calc_distinct_n(chat_no_dialogue_no_react, 2)
    logging.info(f"Distinct 2 for chat with no dialogue and no reaction: {distinct_2_no_dialogue_no_react}")
    distinct_2_no_dialogue = calc_distinct_n(chat_no_dialogue, 2)
    logging.info(f"Distinct 2 for chat with no dialogue: {distinct_2_no_dialogue}")
    distinct_2_dialogue_react = calc_distinct_n(chat_dialogue_react, 2)
    logging.info(f"Distinct 2 for chat with dialogue and reaction: {distinct_2_dialogue_react}")

    # make graphs for distinct 2
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ['Naive', 'No reflections', 'No memory', 'No dialogue and no reaction', 'No dialogue', 'Dialogue and reaction']
    distinct_2 = [distinct_2_naive, distinct_2_no_reflections, distinct_2_no_memory, distinct_2_no_dialogue_no_react, distinct_2_no_dialogue, distinct_2_dialogue_react]

    x = np.arange(len(labels))
    plt.bar(x, distinct_2, color='steelblue')
    plt.xticks(x, labels)
    plt.ylabel("Distinct 2")
    plt.title("Distinct 2 for different approaches")

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add legend
    plt.legend(['Distinct 2'], loc='upper right')

    # make labels readable
    plt.xticks(rotation=45, ha='right')

    # Save the figure
    plt.savefig("thesis_generations/distinct_2.png", bbox_inches='tight')

    # same but only on the first 10 messages
    logging.info("Calculating distinct 2 on the first 10 messages")
    naive_chats_10 = load_chats_from_chat_ids(naive_chat_ids, chat_dir, chat_prefix=naive_prefix, max_messages=10)
    logging.info(f"Loaded {len(naive_chats_10)} naive chats")
    chat_no_reflections_10 = load_chats_from_chat_ids(chat_ids_no_reflections, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    logging.info(f"Loaded {len(chat_no_reflections_10)} chats with no reflections")
    chat_no_memory_10 = load_chats_from_chat_ids(chat_ids_no_memory, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    logging.info(f"Loaded {len(chat_no_memory_10)} chats with no memory")
    chat_no_dialogue_no_react_10 = load_chats_from_chat_ids(chat_ids_no_dialogue_no_react, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    logging.info(f"Loaded {len(chat_no_dialogue_no_react_10)} chats with no dialogue and no reaction")
    chat_no_dialogue_10 = load_chats_from_chat_ids(chat_ids_no_dialogue, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    logging.info(f"Loaded {len(chat_no_dialogue_10)} chats with no dialogue")
    chat_dialogue_react_10 = load_chats_from_chat_ids(chat_ids_dialogue_react, chat_dir, chat_prefix=chat_prefix, max_messages=10)
    logging.info(f"Loaded {len(chat_dialogue_react_10)} chats with dialogue and reaction")

    distinct_2_naive_10 = calc_distinct_n(naive_chats_10, 2)
    logging.info(f"Distinct 2 for naive chat: {distinct_2_naive_10}")
    distinct_2_no_reflections_10 = calc_distinct_n(chat_no_reflections_10, 2)
    logging.info(f"Distinct 2 for chat with no reflections: {distinct_2_no_reflections_10}")
    distinct_2_no_memory_10 = calc_distinct_n(chat_no_memory_10, 2)
    logging.info(f"Distinct 2 for chat with no memory: {distinct_2_no_memory_10}")
    distinct_2_no_dialogue_no_react_10 = calc_distinct_n(chat_no_dialogue_no_react_10, 2)
    logging.info(f"Distinct 2 for chat with no dialogue and no reaction: {distinct_2_no_dialogue_no_react_10}")
    distinct_2_no_dialogue_10 = calc_distinct_n(chat_no_dialogue_10, 2)
    logging.info(f"Distinct 2 for chat with no dialogue: {distinct_2_no_dialogue_10}")
    distinct_2_dialogue_react_10 = calc_distinct_n(chat_dialogue_react_10, 2)
    logging.info(f"Distinct 2 for chat with dialogue and reaction: {distinct_2_dialogue_react_10}")

    # make graphs for distinct 2 first 10
    labels = ['Naive', 'No reflections', 'No memory', 'No dialogue and no reaction', 'No dialogue', 'Dialogue and reaction']

    distinct_2_10 = [distinct_2_naive_10, distinct_2_no_reflections_10, distinct_2_no_memory_10, distinct_2_no_dialogue_no_react_10, distinct_2_no_dialogue_10, distinct_2_dialogue_react_10]

    x = np.arange(len(labels))

    # flush the plot
    plt.clf()

    plt.bar(x, distinct_2_10, color='steelblue')

    plt.xticks(x, labels)

    plt.ylabel("Distinct 2")

    plt.title("Distinct 2 for different approaches on the first 10 messages")

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add legend
    plt.legend(['Distinct 2'], loc='upper right')

    # make labels readable
    plt.xticks(rotation=45, ha='right')

    # Save the figure
    plt.savefig("thesis_generations/distinct_2_10.png", bbox_inches='tight')



    ## both for distinct 3 now
    logging.info("Calculating distinct 3 on complete chats")
    distinct_3_naive = calc_distinct_n(naive_chats, 3)
    logging.info(f"Distinct 3 for naive chat: {distinct_3_naive}")
    distinct_3_no_reflections = calc_distinct_n(chat_no_reflections, 3)
    logging.info(f"Distinct 3 for chat with no reflections: {distinct_3_no_reflections}")
    distinct_3_no_memory = calc_distinct_n(chat_no_memory, 3)
    logging.info(f"Distinct 3 for chat with no memory: {distinct_3_no_memory}")
    distinct_3_no_dialogue_no_react = calc_distinct_n(chat_no_dialogue_no_react, 3)
    logging.info(f"Distinct 3 for chat with no dialogue and no reaction: {distinct_3_no_dialogue_no_react}")
    distinct_3_no_dialogue = calc_distinct_n(chat_no_dialogue, 3)
    logging.info(f"Distinct 3 for chat with no dialogue: {distinct_3_no_dialogue}")
    distinct_3_dialogue_react = calc_distinct_n(chat_dialogue_react, 3)
    logging.info(f"Distinct 3 for chat with dialogue and reaction: {distinct_3_dialogue_react}")

    # make graphs for distinct 3
    labels = ['Naive', 'No reflections', 'No memory', 'No dialogue and no reaction', 'No dialogue', 'Dialogue and reaction']
    distinct_3 = [distinct_3_naive, distinct_3_no_reflections, distinct_3_no_memory, distinct_3_no_dialogue_no_react, distinct_3_no_dialogue, distinct_3_dialogue_react]

    x = np.arange(len(labels))

    plt.bar(x, distinct_3, color='steelblue')

    plt.xticks(x, labels)

    plt.ylabel("Distinct 3")

    plt.title("Distinct 3 for different approaches")

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add legend
    plt.legend(['Distinct 3'], loc='upper right')

    # make labels readable
    plt.xticks(rotation=45, ha='right')

    # Save the figure
    plt.savefig("thesis_generations/distinct_3.png", bbox_inches='tight')

    # same but only on the first 10 messages
    logging.info("Calculating distinct 3 on the first 10 messages")
    distinct_3_naive_10 = calc_distinct_n(naive_chats_10, 3)
    logging.info(f"Distinct 3 for naive chat: {distinct_3_naive_10}")
    distinct_3_no_reflections_10 = calc_distinct_n(chat_no_reflections_10, 3)
    logging.info(f"Distinct 3 for chat with no reflections: {distinct_3_no_reflections_10}")
    distinct_3_no_memory_10 = calc_distinct_n(chat_no_memory_10, 3)
    logging.info(f"Distinct 3 for chat with no memory: {distinct_3_no_memory_10}")
    distinct_3_no_dialogue_no_react_10 = calc_distinct_n(chat_no_dialogue_no_react_10, 3)
    logging.info(f"Distinct 3 for chat with no dialogue and no reaction: {distinct_3_no_dialogue_no_react_10}")
    distinct_3_no_dialogue_10 = calc_distinct_n(chat_no_dialogue_10, 3)
    logging.info(f"Distinct 3 for chat with no dialogue: {distinct_3_no_dialogue_10}")
    distinct_3_dialogue_react_10 = calc_distinct_n(chat_dialogue_react_10, 3)
    logging.info(f"Distinct 3 for chat with dialogue and reaction: {distinct_3_dialogue_react_10}")

    # make graphs for distinct 3 first 10

    labels = ['Naive', 'No reflections', 'No memory', 'No dialogue and no reaction', 'No dialogue', 'Dialogue and reaction']

    distinct_3_10 = [distinct_3_naive_10, distinct_3_no_reflections_10, distinct_3_no_memory_10, distinct_3_no_dialogue_no_react_10, distinct_3_no_dialogue_10, distinct_3_dialogue_react_10]

    x = np.arange(len(labels))

    # flush the plot

    plt.clf()

    plt.bar(x, distinct_3_10, color='steelblue')

    plt.xticks(x, labels)

    plt.ylabel("Distinct 3")

    plt.title("Distinct 3 for different approaches on the first 10 messages")

    # Add grid

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add legend

    plt.legend(['Distinct 3'], loc='upper right')

    # make labels readable

    plt.xticks(rotation=45, ha='right')

    # Save the figure

    plt.savefig("thesis_generations/distinct_3_10.png", bbox_inches='tight')

