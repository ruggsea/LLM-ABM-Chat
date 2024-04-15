import unittest
from unittest.mock import patch
from dialogue_react_agent import load_dialogue_react_fewshots, load_dialogue_act_taxonomy, dialogue_acts, dialogue_react_fewshots, DialogueReactAgent
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

        answer = agent.get_answer(last_messages, extra_context=extra_context, agent_list=agent_list, n_agents=n_agents)

        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()