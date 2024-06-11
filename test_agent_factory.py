import unittest
from unittest.mock import patch
from agent_factory import get_agent_by_topics, DialogueReactAgent, LLMApi, create_groupchat
from llm_engines import ChatgptLLM

class TestAgentFactory(unittest.TestCase):
    def test_get_agent_by_topics_with_persona(self):
        topics = ["Paris"]
        domain = "Clothing"
        name = "John"
        agent_args = {
            "memory_freq": 5,
            "reflections_freq": 15,
            "n_fewshots": 5
        }

        agent = get_agent_by_topics(topics, domain, DialogueReactAgent, name, LLMApi(), **agent_args)

        self.assertIsInstance(agent, DialogueReactAgent)
        self.assertEqual(agent.name, name)
        self.assertIn(name, agent.persona)
        self.assertEqual(agent.memory_freq, 5)
        self.assertEqual(agent.reflections_freq, 15)
        self.assertEqual(agent.n_fewshots, 5)

    def test_get_agent_by_topics_without_persona(self):
        topics = ["technology", "AI"]
        domain = "science"
        agent_args = {
            "memory_freq": 10,
            "reflections_freq": 15,
            "n_fewshots": 5
        }

        with self.assertRaises(ValueError):
            get_agent_by_topics(topics, domain, DialogueReactAgent, None, LLMApi(), **agent_args)

    def test_create_groupchat(self):
        topics_to_include = ["Paris", "Mobile Phones", "Syrian War"]
        n_agents = 3
        agent_args = {
            "memory_freq": 5,
            "reflections_freq": 15,
            "n_fewshots": 5
        }

        agents = create_groupchat(topics_to_include, n_agents, DialogueReactAgent, LLMApi(), **agent_args)

        self.assertEqual(len(agents), n_agents)
        for i, agent in enumerate(agents):
            self.assertIsInstance(agent, DialogueReactAgent)
            self.assertIn(agent.name, agent.persona)
            self.assertEqual(agent.memory_freq, 5)
            self.assertEqual(agent.reflections_freq, 15)
            self.assertEqual(agent.n_fewshots, 5)
            
    def test_create_groupchat_with_empty_topics(self):
        n_agents = 5
        agent_args = {
            "memory_freq": 5,
            "reflections_freq": 15,
            "n_fewshots": 5
        }

        agents = create_groupchat([], n_agents, DialogueReactAgent, LLMApi(), **agent_args)

        self.assertEqual(len(agents), n_agents)
        for i, agent in enumerate(agents):
            self.assertIsInstance(agent, DialogueReactAgent)
            self.assertIn(agent.name, agent.persona)
            self.assertEqual(agent.memory_freq, 5)
            self.assertEqual(agent.reflections_freq, 15)
            self.assertEqual(agent.n_fewshots, 5)
if __name__ == '__main__':
    unittest.main()