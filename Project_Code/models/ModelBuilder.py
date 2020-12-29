import pickle
import ssl

import numpy as np
import pandas as pd

from models.MarkovModel import MarkovModel

__author__ = "Isaac Taylor"


class ModelBuilder(object):
    __password_data: pd.DataFrame
    __markov: MarkovModel
    # stores lowest non zero password probability
    __min_prob: float = 0.00000001
    __FILES = ['../../../Datasets/000webhost.txt', '../../../Datasets/ignis-1M.txt', '../../../Datasets/myspace.txt',
               '../../../Datasets/hotmail.txt']
    __STORED_DATA = '../../../Datasets/combined-data.csv'
    __URL = 'https://drive.google.com/file/d/15Z_GFFqDdH0jnf3t7APB3_49cuofLK2v/view?usp=sharing'
    __STORED_MODEL = 'datastore/markov.pkl'

    def __init__(self, use_stored: bool = True, git=True):
        if use_stored:
            try:
                # opening existing markov model data
                self.__markov = self.load_model(self.__STORED_MODEL)
                if git:
                    path = 'https://drive.google.com/uc?export=download&id=' + self.__URL.split('/')[-2]
                    self.__password_data = pd.read_csv(path)
                else:
                    self.__password_data = pd.read_csv(self.__STORED_DATA)
                self.__min_prob = min([x for x in self.__password_data['probability'] if x != 0])
            except FileNotFoundError:
                self.__markov = MarkovModel('   ', 2)
                self.__password_data = pd.DataFrame()
        else:
            # loading passwords and randomizing their order - 1495126 passwords in total
            self.__password_data = pd.concat((pd.read_csv(r'' + f, header=None, sep=" ", error_bad_lines=False)
                                              for f in self.__FILES)).sample(frac=1).reset_index(drop=True).astype(str)
            self.__password_data.columns = ['password']
            # string of space separated passwords
            passwords = ''.join(password + " " for password in self.__password_data['password'])
            # adding length as a feature
            self.__password_data['length'] = [len(x) for x in self.__password_data['password']]
            # adding number of upper case characters as a feature
            self.__password_data['upper_case'] = [sum(map(str.isupper, x)) for x in self.__password_data['password']]
            # adding number of lower case characters as a feature
            self.__password_data['lower_case'] = [sum(map(str.islower, x)) for x in self.__password_data['password']]
            # adding number of digits as a feature
            self.__password_data['numbers'] = [sum(map(str.isdigit, x)) for x in self.__password_data['password']]
            # adding number of special characters as a feature
            self.__password_data['special_chars'] = [sum(1 for i in x if not i.isalnum())
                                                     for x in self.__password_data['password']]
            # adding number of vowels as a feature
            self.__password_data['vowels'] = [sum(1 for i in x if i in set('aeiouAEIOU'))
                                              for x in self.__password_data['password']]
            # building markov model of order 2
            self.__markov = MarkovModel(passwords, 2)
            # assigning probabilities to passwords
            self.__password_data['probability'] = [self.__markov.ngram_prob(x) for x in
                                                   self.__password_data['password']]
            # making passwords with probability 0 a very small value instead
            self.__min_prob = min([x for x in self.__password_data['probability'] if x != 0])
            self.__password_data['probability'] = [x if x > 0 else (self.__min_prob * 10 ** -1) for x in
                                                   self.__password_data['probability']]

            # better representation of low probabilities
            self.__password_data['log2_prob'] = [np.log2(x) for x in
                                                 self.__password_data['probability']]
            # sorting passwords by probability: most likely appear first
            self.__password_data.sort_values(['probability'], inplace=True, ascending=False)
            # removing duplicate rows
            self.__password_data.drop_duplicates(inplace=True)
            # reindexing rows from 0 since they were randomly shuffled
            self.__password_data.reset_index(inplace=True)
            self.__password_data.drop('index', axis=1, inplace=True)
            # storing results
            self.__password_data.to_csv(self.__STORED_DATA, index=False)
            self.save_model(self.__markov, self.__STORED_MODEL)

    @property
    def foo(self):
        if len(self.__password_data) == 0:
            #lazy initialization
            self.__markov = self.load_model(self.__STORED_MODEL)
            path = 'https://drive.google.com/uc?export=download&id=' + self.__URL.split('/')[-2]
            self.__password_data = pd.read_csv(path)
        return True

    def get_pwd_data(self) -> pd.DataFrame:
        return self.__password_data

    def insert_password(self, pwd: str):
        # insert password into dataframe
        prob = self.__markov.ngram_prob(pwd)
        if prob <= 0:
            prob = self.__min_prob
        row = [pwd, len(pwd), sum([sum(map(str.isupper, x)) for x in pwd]),
               sum([sum(map(str.islower, x)) for x in pwd]),
               sum([sum(map(str.isdigit, x)) for x in pwd]),
               sum([sum(1 for i in x if not i.isalnum()) for x in pwd]),
               sum([sum(1 for i in x if i in set('aeiouAEIOU'))
                    for x in pwd]), prob, np.log2(prob),0]
        self.__password_data['tries'] = [i + 1 for i in range(len(self.__password_data))]
        temp = pd.DataFrame([row], columns=self.__password_data.columns)
        self.__password_data = self.__password_data.append(temp)
        # sorting passwords by probability: most likely appear first
        self.__password_data.sort_values(['probability'], inplace=True, ascending=False)
        # removing duplicate rows
        self.__password_data.drop_duplicates(inplace=True)
        # fixing tries
        self.__password_data['tries'] = [i + 1 for i in range(len(self.__password_data))]

    def predict_tries_ok(self, pwd: str) -> int:
        # model where unlikely passwords were considered

        prob = self.__markov.ngram_prob(pwd)
        if prob <= 0 or np.isnan(prob):
            prob = self.__min_prob
        log2_prob = np.log2(prob)
        poly_2 = int(-1.361021e+06 + -7.556119e+04 * log2_prob +
                     -4.919591e+02 * (log2_prob ** 2))

        return poly_2

    def predict_tries_improved(self, pwd: str) -> int:
        # model where only likely passwords were considered
        prob = self.__markov.ngram_prob(pwd)
        if prob <= 0 or np.isnan(prob):
            prob = self.__min_prob
        log2_prob = np.log2(prob)
        poly_4 = int(-391522.073409 + 20015.215713 * log2_prob +
                     2713.500203 * (log2_prob ** 2) + 43.268947 * (log2_prob ** 3)
                     + 0.198609 * (log2_prob ** 4))

        return poly_4

    def search(self, pwd: str) -> dict:
        # search data for password
        i = 0
        pwds = self.__password_data['password'].values
        for x in pwds:
            if x == pwd:
                return {'pwd': pwd, 'found': 'YES', 'predicted_tries': self.predict_tries_improved(pwd),
                        'actual_tries': i + 1}
            i += 1
        return {'pwd': pwd, 'found': 'NO', 'predicted_tries': self.predict_tries_improved(pwd), 'actual_tries': i + 1}

    def load_model(self, filename: str) -> MarkovModel:
        # loads a model
        model: MarkovModel
        with open(filename, 'rb') as input:
            model = pickle.load(input)
        return model

    def save_model(self, obj: MarkovModel, filename: str):
        # saves an existing model
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


