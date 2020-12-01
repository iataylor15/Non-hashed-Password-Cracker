from _collections import defaultdict
import numpy as np

__author__ = "Isaac Taylor"


class MarkovModel(object):
    # store text
    __text: list
    # order of the model
    __order: int
    # store number of occurrences of kgram in text
    __kgram_count: defaultdict
    # store transitions
    __transitions: defaultdict
    # unique list of characters
    __unique_chars: defaultdict

    def __init__(self, text: str, k: int):
        # creating a markov model of order k
        self.__order = k
        self.__transitions = defaultdict(float)
        self.__kgram_count = defaultdict(int)
        self.__unique_chars = list(set(list(text)))
        n = len(text)
        # adding first k chars of text to end of text
        text += text[:k]

        for i in range(n):
            self.__transitions[text[i:i + k], text[i + k]] += 1.0
            self.__kgram_count[text[i:i + k]] += 1
        return

    def order(self) -> int:
        # returns the order k of markov model
        return self.__order

    def freq_a(self, kgram: str) -> int:
        # returns the number of occurrences of kgram in text
        if self.is_order(kgram):
            return self.__kgram_count[kgram]

    def freq_b(self, kgram: str, c) -> int:
        # returns the number of times that character c follows kgram
        if self.is_order(kgram):
            return self.__transitions[kgram, c]

    def rand(self, kgram: str):
        # returns random character following given kgram
        if self.is_order(kgram):
            total = float(self.freq_a(kgram))
            # creating a random generator
            rand_gen = np.random.Generator(np.random.PCG64())
            return rand_gen.choice(self.__unique_chars, 1, p=np.array([self.__transitions[kgram, x]
                                                                       for x in self.__unique_chars]) / total)

    def gen(self, kgram: str, T: int) -> str:
        # return a string of length T characters by simulating a trajectory through the Markov chain
        result = ''
        if self.is_order(kgram):
            for i in range(T):
                random_char = self.rand(kgram)[0]
                kgram = kgram[1:] + random_char
                result += random_char
        return result

    def ngram_prob(self, ngram: str) -> float:
        # returns probability of a ngram ex. when models order is 2:
        # P(password) = P(pa)P(s|pa)P(s|as)P(w|ss)P(o|sw)P(r|wo)P(d|or)
        k = self.__order
        n = len(ngram)
        prob = 0.0
        if k < n:
            prob = float(self.freq_a(ngram[0:k])) / float(sum(self.__kgram_count.values()))
            for i in range(n - k):
                prob *= (float(self.freq_b(ngram[i:i + k], ngram[i + k])) / float(self.freq_a(ngram[i:i + k])))
        return prob

    def is_order(self, kgram):
        # checks to see if kgram is of order k
        return len(kgram) == self.__order
