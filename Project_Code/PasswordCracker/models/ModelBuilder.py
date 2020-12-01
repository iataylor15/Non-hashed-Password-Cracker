from models.MarkovModel import MarkovModel
import pandas as pd
import numpy as np
import sys
import pickle
import time
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse

__author__ = "Isaac Taylor"


class ModelBuilder:
    __password_data: pd.DataFrame
    __markov: MarkovModel
    # stores least likely password probability
    __min_prob: float
    __FILES = ['../../../Datasets/000webhost.txt', '../../../Datasets/ignis-1M.txt', '../../../Datasets/myspace.txt',
               '../../../Datasets/hotmail.txt']
    __STORED_DATA = '../../../Datasets/combined-data.csv'
    __STORED_MODEL = '../datastore/markov.pkl'

    def __init__(self, use_stored: bool = False):
        if use_stored:
            try:
                # opening existing markov model data
                self.__markov = self.load_model(self.__STORED_MODEL)
                self.__password_data = pd.read_csv(self.__STORED_DATA)
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
            min_prob = min([x for x in self.__password_data['probability'] if x != 0])
            self.__min_prob = min_prob
            self.__password_data['probability'] = [x if x > 0 else (min_prob * 10 ** -1) for x in
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

    def get_pwd_data(self) -> pd.DataFrame:
        return self.__password_data

    def insert_password(self, pwd: str):
        prob = self.__markov.ngram_prob(pwd)
        row = [pwd, len(pwd), sum([sum(map(str.isupper, x)) for x in pwd]),
               sum([sum(map(str.islower, x)) for x in pwd]),
               sum([sum(map(str.isdigit, x)) for x in pwd]),
               sum([sum(1 for i in x if not i.isalnum()) for x in pwd]),
               sum([sum(1 for i in x if i in set('aeiouAEIOU'))
                    for x in pwd]), prob, np.log2(prob), 0]

        temp = pd.DataFrame([row], columns=self.__password_data.columns)
        self.__password_data = self.__password_data.append(temp)
        # sorting passwords by probability: most likely appear first
        self.__password_data.sort_values(['probability'], inplace=True, ascending=False)
        # removing duplicate rows
        self.__password_data.drop_duplicates(inplace=True)
        # fixing tries
        self.__password_data['tries'] = [i + 1 for i in range(len(self.__password_data))]

    def predict_tries(self, pwd: str) -> int:
        df1 = self.__password_data
        df1['tries'] = [i + 1 for i in range(len(df1))]
        prob = self.__markov.ngram_prob(pwd)
        if prob < 0:
            prob = self.__min_prob * 10 ** -1
        log2_prob = np.log2(prob)
        poly_5 = smf.ols(formula='tries ~ 1 + log2_prob + I(log2_prob ** 2.0) + I(log2_prob ** 3.0) + '
                                 'I(log2_prob ** 4.0) + I(log2_prob ** 5.0)', data=df1).fit()
        result = int(poly_5.params['Intercept'] + (poly_5.params['log2_prob']) * log2_prob)
        if result < 0:
            return 1
        else:
            return result

    def search(self, pwd: str):
        i = 0
        pwds = self.__password_data['password'].values
        for x in pwds:
            if x == pwd:
                return {'found': True, 'tries': i + 1}
            i += 1
        return {'found': False, 'tries': i + 1}

    def load_model(self, filename: str) -> MarkovModel:
        model: MarkovModel
        with open(filename, 'rb') as input:
            model = pickle.load(input)
        return model

    def save_model(self, obj: MarkovModel, filename: str):
        # saves an existing model
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
