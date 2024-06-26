import unittest
from chat_eval import calc_distinct_n

class TestChatEval(unittest.TestCase):
    def test_calc_distinct_n_single_n1(self):
        chat_history = "Hello, how are you?\nI'm good, thanks!"
        expected_result = 7
        self.assertEqual(calc_distinct_n(chat_history, n=1), expected_result)

    def test_calc_distinct_n_single_n2(self):
        chat_history = "Hello, how are you?\nI'm good, thanks!"
        expected_result = 6
        self.assertEqual(calc_distinct_n(chat_history, n=2), expected_result)

    def test_calc_distinct_n_multiple_n1(self):
        chat_history = ["Hello, how are you?", "I'm good, thanks!"]
        expected_result = (4+3)/2
        self.assertEqual(calc_distinct_n(chat_history, n=1), expected_result)

    def test_calc_distinct_n_multiple_n2(self):
        chat_history = ["Hello, how are you?", "I'm good, thanks!"]
        expected_result = (3+2)/2
        self.assertEqual(calc_distinct_n(chat_history, n=2), expected_result)

    def test_calc_distinct_n_empty(self):
        chat_history = ""
        expected_result = 0
        self.assertEqual(calc_distinct_n(chat_history, n=1), expected_result)

    def test_calc_distinct_n_special_characters(self):
        chat_history = "Hello, how are you? I'm good, thanks!"
        expected_result = 7
        self.assertEqual(calc_distinct_n(chat_history, n=1), expected_result)

    def test_calc_distinct_n_repeated_words(self):
        chat_history = "Hello, hello, how are you?"
        expected_result = 4
        self.assertEqual(calc_distinct_n(chat_history, n=1), expected_result)

    def test_calc_distinct_n_repeated_ngrams(self):
        chat_history = "Hello, how are you? Hello, how are you?"
        expected_result = 4
        self.assertEqual(calc_distinct_n(chat_history, n=1), expected_result)

    def test_calc_distinct_n_repeated_different_ngrams(self):
        chat_history = ["Hello, how are you?", "Hello, how are you?"]
        expected_result = 4
        self.assertEqual(calc_distinct_n(chat_history, n=1), expected_result)
        
    # now with n=3
    
    def test_calc_distinct_n_single_n3(self):
        chat_history = "Ciao mi chiamo roberto e mi piace il gelato"
        expected_result = 7
        self.assertEqual(calc_distinct_n(chat_history, n=3), expected_result)
    
    # now with n=4
    
    def test_calc_distinct_n_single_n4(self):
        chat_history = "Ciao mi chiamo roberto e mi piace il gelato"
        expected_result = 6
        self.assertEqual(calc_distinct_n(chat_history, n=4), expected_result)
    
    # again but with repeated words
    
    def test_calc_distinct_n_repeated_words_n3(self):
        chat_history = "Ciao mi chiamo roberto e mi chiamo roberto"
        expected_result = 5
        self.assertEqual(calc_distinct_n(chat_history, n=3), expected_result)
        
        
if __name__ == '__main__':
    unittest.main()

