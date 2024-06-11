import unittest
from unittest.mock import patch
from dialogue_react_agent import load_dialogue_react_fewshots, load_dialogue_act_taxonomy, DialogueReactAgent
from llm_engines import LLMApi

class TestDialogueReactAgent(unittest.TestCase):
    def test_load_dialogue_react_fewshots(self):
        file_path = "fewshots_mpc/generated_dialogue_reacts.jsonl"
        prompts = load_dialogue_react_fewshots(file_path)
        self.assertIsInstance(prompts, list)
        self.assertGreater(len(prompts), 0)

    def test_load_dialogue_act_taxonomy(self):
        file_path = "prompts/dialogue_taxonomy.txt"
        acts = load_dialogue_act_taxonomy(file_path)
        self.assertIsInstance(acts, list)
        self.assertGreater(len(acts), 0)

    @patch('logging.basicConfig')
    def test_main(self, mock_logging):
        name = "John"
        llm = LLMApi()
        persona = "I am a persona"
        memory_freq = 0.5
        reflections_freq = 0.3
        n_fewshots = 5

        agent = DialogueReactAgent(name, llm, persona, memory_freq, reflections_freq, n_fewshots)

        self.assertEqual(agent.persona, persona)
        self.assertEqual(agent.memory_freq, memory_freq)
        self.assertEqual(agent.reflections_freq, reflections_freq)
        self.assertEqual(agent.n_fewshots, n_fewshots)

    def test_get_answer(self):
        last_messages = ["Hello", "How are you?"]
        extra_context = "Some extra context"
        agent_list = ["Agent1", "Agent2"]
        n_agents = 2

        name = "John"
        llm = LLMApi()
        persona = "I am a persona"
        memory_freq = 0.5
        reflections_freq = 0.3
        n_fewshots = 5

        agent = DialogueReactAgent(name, llm, persona, memory_freq, reflections_freq, n_fewshots)

        answer = agent.get_answer(last_messages, extra_context=extra_context, agent_list=agent_list, n_agents=n_agents, turn_count=4)

        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")
        # Add more assertions as needed

    def test_gen_memories(self):
        last_messages = [
            (1, "Mark", "I like discussing about 18th century german philosophy."),
            (2, "Alex", "Oh really? I really think that in his seminal work, Kant was really onto something."),
            (3, "Mark", "I disagree. I think that Hegel's dialectical method is more relevant than ever.")
        ]
        n_memories = 3
        agent_list = ["Mark", "Alex"]
        n_agents = 2
        turn_count = 4

        name = "Mark"
        llm = LLMApi()
        persona = "I am a persona"
        memory_freq = 0.5
        reflections_freq = 0.3
        n_fewshots = 5

        agent = DialogueReactAgent(name, llm, persona, memory_freq, reflections_freq, n_fewshots)

        memories = agent.gen_memories(last_messages, n_memories, agent_list=agent_list, n_agents=n_agents, turn_count=turn_count)

        self.assertIsInstance(memories, list)
        self.assertEqual(len(memories), n_memories)
        for i, memory in enumerate(memories):
            self.assertIsInstance(memory, dict)
            self.assertIn("turn", memory)
            self.assertIn("text", memory)
            self.assertIn("memory_n", memory)
            # check that the memory has the correct index
            self.assertEqual(memory["memory_n"], i+1)
        
            
    def test_gen_reflection(self):
        last_messages = [
            (1, "Mark", "I like discussing about 18th century german philosophy."),
            (2, "Alex", "Oh really? I really think that in his seminal work, Kant was really onto something"),
            (3, "Mark", "I disagree. I think that Hegel's dialectical method is more relevant than ever.")
        ]
        n_memories = 3
        n_reflections = 5
        agent_list = ["Mark", "Alex"]
        n_agents = 2
        turn_count = 4

        name = "Mark"
        llm = LLMApi()
        persona = "I am a persona"
        memory_freq = 5
        reflections_freq = 10
        n_fewshots = 5

        agent = DialogueReactAgent(name, llm, persona, memory_freq, reflections_freq, n_fewshots)
        
        memories=agent.gen_memories(last_messages, n_memories, agent_list=agent_list, n_agents=n_agents, turn_count=turn_count)
        
        agent.save_observations(memories)

        reflections = agent.gen_reflection(n_memories, n_reflections, agent_list=agent_list, n_agents=n_agents, turn_count=turn_count)

        self.assertIsInstance(reflections, list)
        self.assertEqual(len(reflections), n_reflections)
        for i, reflection in enumerate(reflections):
            self.assertIsInstance(reflection, dict)
            self.assertIn("memory_n", reflection)
            self.assertIn("turn", reflection)
            self.assertIn("text", reflection)
            # check that the reflection has the correct index
            self.assertEqual(reflection["memory_n"], agent.memory.count() + i + 1)          
            
         

    def test_save_observations(self):
        observations_list = [
            {"id": 1, "text": "Observation 1"},
            {"id": 2, "text": "Observation 2"},
            {"id": 3, "text": "Observation 3"}
        ]

        name = "John"
        llm = LLMApi()
        persona = "I am a persona"
        memory_freq = 5
        reflections_freq = 10
        n_fewshots = 5

        agent = DialogueReactAgent(name, llm, persona, memory_freq, reflections_freq, n_fewshots)

        agent.save_observations(observations_list)

        # Verify that the observations are saved in memory
        self.assertEqual(agent.memory.count(), len(observations_list))
        total_memories = agent.memory.search("select id, text from txtai")
        for observation in observations_list:
            for memory in total_memories:
                if memory["text"] == observation["text"]:
                    break
            else:
                self.fail(f"Observation {observation} not found in memory")
        

    
    def test_generate_with_memory(self):
        last_messages = [
            (1, "Mark", "I like discussing about 18th century german philosophy."),
            (2, "Alex", "Oh really? I really think that in his seminal work, Kant was really onto something"),
            (3, "Mark", "I disagree. I think that Hegel's dialectical method is more relevant than ever.")
        ]
        extra_context = "Some extra context"
        agent_list = ["Mark", "Alex"]
        n_agents = 2

        name = "Mark"
        llm = LLMApi()
        persona = "I am a persona"
        memory_freq = 5
        reflections_freq = 10
        n_fewshots = 5

        agent = DialogueReactAgent(name, llm, persona, memory_freq, reflections_freq, n_fewshots)
        
        ## generate the memory
        
        memories = agent.gen_memories(last_messages, n_memories=5, agent_list=agent_list, n_agents=n_agents, turn_count=4)
        
        ## save the memory
        
        agent.save_observations(memories)
        
        answer = agent.get_answer(last_messages, extra_context=extra_context, agent_list=agent_list, n_agents=n_agents, turn_count=4)

        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")
        
    def test_run_routines(self):
        turn_count = 5
        chat_history = [
            (1, "Mark", "Hello"),
            (2, "Alex", "Hi Mark, how are you?"),
            (3, "Mark", "I'm good, thanks for asking.")
        ]

        name = "John"
        llm = LLMApi()
        persona = "I am a persona"
        memory_freq = 5
        reflections_freq = 10
        n_fewshots = 5

        agent = DialogueReactAgent(name, llm, persona, memory_freq, reflections_freq, n_fewshots)

        agent.run_routines(turn_count, chat_history, n_agents=2, agent_list=["Mark", "Alex"])

        # Verify that save_observations is called when turn_count is divisible by memory_freq
        if turn_count % memory_freq == 0:
            assert agent.memory.count()==5

        ## turn count is divisible by reflections_freq
        turn_count = 20
        
        print(f"N memory objects before: {agent.memory.count()}")
        agent.run_routines(turn_count, chat_history, n_agents=2, agent_list=["Mark", "Alex"])
        print(f"N memory objects after: {agent.memory.count()}")
        
        if turn_count % reflections_freq == 0:
            assert agent.memory.count()==15

if __name__ == '__main__':
    unittest.main()
