import numpy as np
import string

class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.chain = {}

    def clean_word(self, word):
        return word.translate(str.maketrans('', '', string.punctuation)).lower()

    def train(self, text):
        words = [self.clean_word(w) for w in text.split()]
        for i in range(len(words) - self.order):
            state = tuple(words[i:i + self.order])
            next_word = words[i + self.order]

            if state not in self.chain:
                self.chain[state] = {}

            if next_word not in self.chain[state]:
                self.chain[state][next_word] = 0

            self.chain[state][next_word] += 1

    def predict_next(self, input_text):
        input_words = [self.clean_word(w) for w in input_text.split()]
        state = tuple(input_words[-self.order:])

        if state not in self.chain:
            return None

        next_words = list(self.chain[state].keys())
        counts = list(self.chain[state].values())
        probabilities = [c / sum(counts) for c in counts]

        return np.random.choice(next_words, p=probabilities)


# -------------------------
# 🔧 Main Program
# -------------------------
if __name__ == "__main__":
    sample_text = """
    It is a sunny day. It is a bright and beautiful morning.
    It is going to rain. It is what it is. The sun is shining.
    The weather is nice today. It might rain tomorrow. It is unpredictable.
    """

    mc = MarkovChain(order=2)
    mc.train(sample_text)

    print("🔮 Markov Chain: Next Word Prediction 🔮")
    while True:
        phrase = input("\nEnter a phrase (or 'exit' to quit): ")
        if phrase.lower() == "exit":
            break

        predicted = mc.predict_next(phrase)
        if predicted:
            print(f"Predicted next word: '{predicted}'")
        else:
            print("No prediction available (unseen phrase).")
