import unittest
from chat_eval import calc_distinct_n, calc_llm_as_a_judge_pairwise, calc_perplexity
import numpy as np

class TestChatEval(unittest.TestCase):
    # test if calc_disitnct_n returns the correct value
    # it should give out a vector of (number of unique n-grams)/(total words) for each chat
    def test_calc_distinct_n(self, n=1):
        chat_history = "hello hello ciao"
        expected_result = [2/3]
        self.assertEqual(calc_distinct_n(chat_history, n), expected_result)
    
    # now the same test but with n=2
    def test_calc_distinct_n_n2(self):
        chat_history = "hello hello ciao"
        expected_result = [2/3]
        self.assertEqual(calc_distinct_n(chat_history, 2), expected_result)
    
    # test perpexity calculation
    # it should give out a vector of perplexity values for each chat
    def test_calc_perplexity(self):
        chat_history = "hello hello ciao"
        # assert it is a list of floats
        self.assertIsInstance(calc_perplexity(chat_history), list)
    
    def test_calc_multi_perplexity(self):
        chat_history = ["hello hello ciao", "Hello I'm Mark"]
        # assert it is a list of floats
        self.assertIsInstance(calc_perplexity(chat_history), list)
        # check that perplexity 1 is higher than perplexity 2
        self.assertGreater(calc_perplexity(chat_history)[0], calc_perplexity(chat_history)[1])
    
    def test_calc_llm_as_a_judge_pairwise_single_chat(self):
        chat_history_a = "Hello, how are you?\nI'm good, thanks!"
        chat_history_b = "Hey how are you?\nI'm from Scotland"
        expected_result = ["A"]
        self.assertEqual(calc_llm_as_a_judge_pairwise(chat_history_a, chat_history_b), expected_result)
        
    def test_calc_llm_as_a_judge_pairwise_multiple_chats(self):
        chat_history_a = ["Hello, how are you?\n Not bad", "Do you like soccer?\n Barcellona is the capital of Sweeden"]
        chat_history_b = ["Hi, are you Mark?\n Yes, I speak finnish", "How are you?\n I'm great, thanks!"]
        expected_result = ["A", "B"]
        self.assertEqual(calc_llm_as_a_judge_pairwise(chat_history_a, chat_history_b), expected_result)
        
        
    def test_calc_llm_as_a_judge_pairwise_different_model(self):
        chat_history_a = "Hello, how are you?\n I'm good, thanks!"
        chat_history_b = "Paola: What's your name?\n Eric: I'm fine, thanks!"
        expected_result = ["A"]
        self.assertEqual(calc_llm_as_a_judge_pairwise(chat_history_a, chat_history_b, model="prometheus-2.0", n_consistency=3), expected_result)
        
        
if __name__ == '__main__':
    unittest.main()