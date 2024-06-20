import unittest
from llm_engines import LLMApi, ChatgptLLM
from places_replication import NaiveConversationGeneration, NaiveConversationAgent

class TestNaiveConversationGeneration(unittest.TestCase):
    def setUp(self):
        self.llm = ChatgptLLM()
        self.agent_list = [NaiveConversationAgent("Alice", llm=self.llm), NaiveConversationAgent("Bob", llm=self.llm)]
        self.topics_to_cover = ["Paris", "London"]
        self.conversation_generator = NaiveConversationGeneration(self.agent_list, self.llm, self.topics_to_cover)

    def test_generate_conversation(self):
        conversation = self.conversation_generator.generate_conversation(min_turns=10, start_conversation=True)
        self.assertEqual(len(conversation), 10)
        self.assertIn(conversation[0][1], [agent.name for agent in self.agent_list])
        self.assertIn(conversation[0][2], [conversation_start for conversation_start in self.conversation_generator.conversation_starters])

    def test_print_chat_history(self):
        # Assuming conversation history is not empty
        self.conversation_generator.generate_conversation(min_turns=5, start_conversation=True)
        self.conversation_generator.print_chat_history()
        # No assertion, just checking if it prints without errors

    def test_save_chat_history(self):
        # Assuming conversation history is not empty
        self.conversation_generator.generate_conversation(min_turns=5, start_conversation=True)
        self.conversation_generator.save_chat_history()
        # No assertion, just checking if it saves without errors

    def test_dump_chat(self):
        # Assuming conversation history is not empty
        self.conversation_generator.generate_conversation(min_turns=5, start_conversation=True)
        self.conversation_generator.dump_chat()
        # No assertion, just checking if it dumps without errors

if __name__ == '__main__':
    unittest.main()