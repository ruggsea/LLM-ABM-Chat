


import unittest
from llm_engines import LLMApi
from chat_llm import Agent, ChatThread


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.llm = LLMApi()
        self.agent = Agent("John", self.llm, interests=["sports"], behavior=["friendly"])

    def test_get_name(self):
        self.assertEqual(self.agent.get_name(), "John")

    def test_get_answer(self):
        last_messages = [("Alice", "Hello"), ("Bob", "How are you?")]
        expected_prompt = "The last messages were: Alice: Hello\nBob: How are you?. What do you answer?"
        response = self.agent.get_answer(last_messages)
        # make sure the answer is properly formatted
        self.assertTrue(response)




class TestChatThread(unittest.TestCase):
    def setUp(self):
        self.agent_list = [Agent("John", LLMApi(), interests=["sports"], behavior=["friendly"]), Agent("Alice", LLMApi(), interests=["music"], behavior=["friendly"])
        ]
        self.chat_goal = 'Test chat goal'
        self.chat_thread = ChatThread(self.agent_list, self.chat_goal)

    def test_pick_random_agent(self):
        random_agent = self.chat_thread.pick_random_agent()
        self.assertIn(random_agent, self.agent_list)

    def test_start_conversation(self):
        start_message = self.chat_thread.start_conversation()
        self.assertIn(start_message, self.chat_thread.chat_history)

    def test_get_chat_answer(self):
        last_messages = [("John", "Hello"), ("Alice", "How are you?")]
        agent = self.agent_list[0]
        self.assertTrue(self.chat_thread.get_chat_answer(last_messages, agent))

    def test_run_chat(self):
        max_turns = 3
        chat_history = self.chat_thread.run_chat(max_turns)
        self.assertEqual(len(chat_history), max_turns)


if __name__ == '__main__':
    unittest.main()
    
