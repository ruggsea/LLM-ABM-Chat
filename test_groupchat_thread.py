import unittest
from llm_engines import LLMApi, ChatgptLLM
from groupchat_thread import GroupchatThread
from agent_factory import create_groupchat
from dialogue_react_agent import DialogueReactAgent

class TestGroupchatThread(unittest.TestCase):
    def setUp(self):
        self.llm = ChatgptLLM()
        self.agent_args = {
            "memory_freq": 5,
            "reflections_freq": 15,
            "n_fewshots": 5
        }
        self.agent_list = create_groupchat(topics_to_include=["Paris"], n_agents=2, neutral_llm=self.llm, agent_type=DialogueReactAgent, **self.agent_args)
        self.groupchat_thread = GroupchatThread(self.agent_list, self.llm)

    def test_start_conversation(self):
        self.groupchat_thread.start_conversation()
        self.assertEqual(len(self.groupchat_thread.chat_history), 1)
        self.assertEqual(self.groupchat_thread.turn, 1)
    
    def test_pick_random_agent(self):
        random_agent = self.groupchat_thread.pick_random_agent()
        self.assertIn(random_agent, self.agent_list)

    def test_get_chat_answer(self):
        last_messages = [(1, self.groupchat_thread.agent_list[0].name, "Hello!"), (2, self.groupchat_thread.agent_list[1].name, "Hi!")]
        agent = self.agent_list[0]
        answer = self.groupchat_thread.get_chat_answer(last_messages, agent)
        self.assertIsInstance(answer, str)

    def test_render_last_message(self):
        self.groupchat_thread.start_conversation()
        self.groupchat_thread.render_last_message()

    def test_evaluate_chat(self):
        self.groupchat_thread.start_conversation()
        self.groupchat_thread.evaluate_chat()

    def test_dump_chat(self):
        self.groupchat_thread.start_conversation()
        self.groupchat_thread.dump_chat()

    def test_run_chat(self):
        chat_history = self.groupchat_thread.run_chat(max_turns=3)
        self.assertEqual(len(chat_history), 3)

if __name__ == '__main__':
    unittest.main()