import unittest
from llm_engines import LLMApi
from chat_llm import MemoryAgent

class TestMemoryAgent(unittest.TestCase):
    def setUp(self):
        self.llmapi = LLMApi()
        self.memory_agent = MemoryAgent("John", self.llmapi)

    def test_save_observations(self):
        observations = ["Observation 1", "Observation 2", "Observation 3", "Observation 4", "Observation 5"]
        self.memory_agent.save_observations(observations)
        self.assertTrue(self.memory_agent.memory.isdense())

if __name__ == '__main__':
    unittest.main()