import unittest
from llm_engines import LLM, LLMApi, ChatgptLLM

class TestLLM(unittest.TestCase):
    def setUp(self):
        self.llm = LLM()

    def test_set_system_prompt(self):
        self.llm.set_system_prompt('test prompt')
        self.assertEqual(self.llm.get_system_prompt(), 'test prompt')

    def test_set_history(self):
        self.llm.set_history([{'role': 'system', 'content': 'test history'}])
        self.assertEqual(self.llm.get_history(), [{'role': 'system', 'content': 'test history'}])

class TestLLMApi(unittest.TestCase):
    def setUp(self):
        self.llmapi = LLMApi()
        
    def test_generate_response(self):
        self.assertTrue(self.llmapi.generate_response('test prompt'))
        # check that history was updated by comparing length of history before and after
        self.assertEqual(len(self.llmapi.get_history()), 2)        

class TestChatgptLLM(unittest.TestCase):
    def setUp(self):
        self.chatgpt_llm = ChatgptLLM()

    def test_set_system_prompt(self):
        self.chatgpt_llm.set_system_prompt('test user prompt')
        self.assertEqual(self.chatgpt_llm.get_system_prompt(), 'test user prompt')

    def test_generate_response(self):
        self.assertTrue(self.chatgpt_llm.generate_response('test prompt'))
        # check that history was updated by comparing length of history before and after
        self.assertEqual(len(self.chatgpt_llm.get_history()), 2)

if __name__ == '__main__':
    unittest.main()
