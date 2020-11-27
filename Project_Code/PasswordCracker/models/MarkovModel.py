from _collections import defaultdict
import numpy as np


class MarkovModel:
    # store text
    __text: list
    # order of the model
    __order: int
    # store number of occurrences of kgram in text
    __gram_count: defaultdict
    # store probabilities of a  transition
    __probabilities: defaultdict
    # unique list of characters
    __unique_chars: defaultdict

    def __init__(self, text: str, k: int):
        # creating a markov model of order k
        self.__order = k
        self.__probabilities = defaultdict(float)
        self.__gram_count = defaultdict(int)
        self.__unique_chars = list(set(list(text)))
        N = len(text)
        # adding first k chars of text to end of text
        text += text[:k]

        for i in range(N):
            self.__probabilities[text[i:i + k], text[i + k]] += 1.0
            self.__gram_count[text[i:i + k]] += 1
        return

    def order(self) -> int:
        # returns the order k of markov model
        return self.__order

    def freq(self, kgram: str) -> int:
        # returns the number of occurrences of kgram in text
        if self.is_order(kgram):
            return self.__gram_count[kgram]

    def freq(self, kgram: str, c) -> int:
        # returns the number of times that character c follows kgram
        if self.is_order(kgram):
            return self.__gram_count[kgram, c]

    def rand(self, kgram: str):
        # returns random character following given kgram
        if self.is_order(kgram):
            total = sum([self.__probabilities[kgram, x] for x in self.__unique_chars])
            # creating a random generator
            rand_gen = np.random.Generator(np.random.PCG64())
            return rand_gen.choice(self.__unique_chars, 1, p=np.array([self.__probabilities[kgram, x]
                                                                       for x in self.__unique_chars]) / total)

    def gen(self, kgram: str, T: int) -> str:
        # return a string of length T characters by simulating a trajectory through the Markov chain
        if self.is_order(kgram):
            result = ''
            for i in range(T):
                random_char = self.rand(kgram)[0]
                kgram = kgram[1:] + random_char
                result += random_char
        return result

    def is_order(self, kgram):
        # checks to see if kgram is of order k
        return len(kgram) == self.__order


# x = np.array([[1, 23, 4, 56], [2, 4, 6, 8]])
# a = np.array([1])
# b = np.array([2])
y = 'A green apple and brown dog; ants ate all of the apples.'
# print(x[a, b])
# print(y)

model = MarkovModel(y, 1)
print(('a'+model.gen('a', 3)).split(' ', maxsplit=1)[0])
