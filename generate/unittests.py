import unittest
from markov_chain import generate_chain, generate_text, main

class TestMarkovChain(unittest.TestCase):
    
    def test_generate_chain_order_one(self):
        text = "I saw a cat. I saw a dog."
        chain = generate_chain(text, 1)
        self.assertEqual(chain, {("I",): ["saw", "saw"], ("saw",): ["a", "a"], ("a",): ["cat.", "dog."], ("cat.",): ["I"]})

    def test_generate_text_order_one(self):
        chain = {("I",): ["saw", "saw"], ("saw",): ["a", "a"], ("a",): ["cat.", "dog."], ("cat.",): ["I"]}
        text = generate_text(chain, 6, "I")
        self.assertTrue(text.startswith("I saw") and text.count(' ') == 5)

    def test_generate_text_empty_chain_key(self):
        chain = {("I",): [], ("saw",): ["a", "a"], ("a",): ["cat.", "dog."], ("cat.",): ["I"]}
        text = generate_text(chain, 6, "I")
        # The chain for the seed "I" is empty, so the first word will be chosen randomly
        self.assertTrue(text.startswith(("I I", "I saw", "I a", "I cat.")) and text.count(' ') == 5)


    def test_generate_chain_order_two(self):
        text = "I saw a cat. I saw a dog."
        chain = generate_chain(text, 2)
        self.assertEqual(chain, {("I", "saw"): ["a", "a"], ("saw", "a"): ["cat.", "dog."], ("a", "cat."): ["I"], ("cat.", "I"): ["saw"]})

    def test_generate_text_order_two(self):
        chain = {("I", "saw"): ["a", "a"], ("saw", "a"): ["cat.", "dog."], ("a", "cat."): ["I"], ("cat.", "I"): ["saw"]}
        text = generate_text(chain, 6, "I saw")
        self.assertTrue(text.startswith("I saw a") and text.count(' ') == 5)

if __name__ == "__main__":
    unittest.main()
