import pickle
import pymorphy2  # python -m pip install pymorphy2 #python -m pip install gensim
import Levenshtein  # python -m pip install python-levenshtein
import stringdist  # python -m pip install stringdist
from scipy.special import softmax
from gensim.models import FastText

import pandas as pd
import numpy as np

from tqdm import tqdm
import re
import os

class SpellChecker:

    def __init__(self):
        print(os.path.dirname(__file__))
        self.total_word_dict = pickle.load(open('../data/spellchecker_prototype_v2/total_word_dict.pickle', 'rb'))
        # self.tomsk_no_stop = pickle.load(open(r'spell_checker/data/tomsk_no_stop.pickle', 'rb'))

        # self.total_word_dict = self.total_word_dict1 + self.tomsk_no_stop
        self.wordcount = pickle.load(open('../data/spellchecker_prototype_v2/wordcount.pickle', 'rb'))

        # self.model = FastText.load(r'spell_checker/models/cbow_model_tmsk_all.model', mmap='r')
        # self.model = FastText.load(r'spell_checker/models/cbow_model_tmsk_3.model', mmap='r')
        self.model = FastText.load('../data/spellchecker_prototype_v2/models/cbow_model_new.model', mmap='r')
        # self.exp = pickle.load(open(r'spell_checker/data/exp.pickle', 'rb'))

    def normalize_word(self, word):
        morph = pymorphy2.MorphAnalyzer()
        # return word if word in self.exp else self.morph.normal_forms(word)[0]
        return morph.parse(word)[0].normal_form

    def check_for_correction(self, word):
        # if (word in self.total_word_dict) or (word in self.exp):
        if (word in self.total_word_dict):
            return word
        else:
            return False

    def get_double_word(self, word):
        morph = pymorphy2.MorphAnalyzer()
        for i in range(2, len(word)):
            part_a = morph.parse(word[:i])[0].normal_form
            part_b = morph.parse(word[i:len(word)])[0].normal_form
            if part_a in self.total_word_dict and part_b in self.total_word_dict:
                return [part_a, part_b]
        else:
            return []

    def predict_fast_text(self, word):

        if word in self.model.wv:

            top_words = [w[0] for w in self.model.wv.most_similar(positive=[word], topn=1000000)]
            top_scores = [w[1] for w in self.model.wv.most_similar(positive=[word], topn=1000000)]

        else:
            print(f'No word {word} in model')
        return top_words, top_scores

    def apply_fast_text(self, word, topn):
        ft_words, ft_scores = self.predict_fast_text(word)

        models_distances = pd.DataFrame(data=ft_scores, index=ft_words, columns=['FastText'])
        models_distances['mean'] = models_distances.mean(axis=1)
        return ft_words, ft_scores, models_distances

    def calc_levenshtein(self, a, b, distance='dlevenshtein'):

        if distance == 'levenshtein':
            return Levenshtein.distance(a, b)

        elif distance == 'dlevenshtein':
            return stringdist.rdlevenshtein(a, b)

    def apply_levenshtein(self, word, smax=False):
        words = [word] * len(self.total_word_dict)
        if smax:
            levenshtein_scores = softmax(np.array(list(map(self.calc_levenshtein, words, self.total_word_dict))))
        else:
            levenshtein_scores = np.array(list(map(self.calc_levenshtein, words, self.total_word_dict)))

        lev_min_score = np.array(levenshtein_scores).min()
        index_list = list([np.where(levenshtein_scores == lev_min_score)])[0][0]
        lev_words = [self.total_word_dict[idx] for idx in index_list]

        return lev_words

    def apply_levenshtein_and_model(self, word, topn):
        ft_words, ft_scores, models_distances = self.apply_fast_text(word, self.model)

        ft_top_words, ft_top_scores = ft_words[:topn], ft_scores[:topn]
        lev_words = self.apply_levenshtein(word, self.total_word_dict)



        wordcount_scores = [self.wordcount[self.wordcount['word'] == ft_word]['counter'].values[0]\
                                if not self.wordcount[self.wordcount['word'] == ft_word].empty else 0\
                            for ft_word in ft_top_words]
        probas_lev = []
        for lw in lev_words:
            if not self.wordcount[self.wordcount['word'] == lw].empty:
                probas_lev.append(self.wordcount[self.wordcount['word'] == lw]\
                                      ['counter'].values[0] / len(self.wordcount))
            else:
                probas_lev.append(0)

        word_list = lev_words
        word_list = list(set(word_list + [self.normalize_word(w) for w in word_list]))

        try:
            # corrected_word = models_distances['mean'].loc[word_list].argmax()
            corrected_word = models_distances.loc[word_list].iloc[models_distances['mean'].loc[word_list].argmax()].name
            print(f'try {corrected_word}',)

        except (KeyError, ValueError) as e:
            print(f'Fast Text Exception for {word}', )
            corrected_word = word_list[np.array(probas_lev).argmax()]

        method_word = corrected_word
        return method_word

    def choose_between_double_and_single_words(self, double_word, method_word):
        result_words = [None, None]
        result_words[0] = ' '.join(double_word)
        result_words[1] = method_word
        result_probas = [0, 0]

        if not self.wordcount[self.wordcount['word'] == method_word].empty:
            result_probas[1] = self.wordcount[self.wordcount['word'] == method_word]\
                                   ['counter'].values[0] / len(self.wordcount)

        double_probas = [0, 0]
        if len(double_word) > 1:
            for j, d_word in enumerate(double_word):
                if not self.wordcount[self.wordcount['word'] == d_word].empty:
                    double_probas[j] = self.wordcount[self.wordcount['word'] == d_word]\
                                           ['counter'].values[0] / len(self.wordcount)
            result_probas[0] = np.mean(double_probas)

            for d_word in double_word:
                if d_word not in self.total_word_dict:
                    result_probas[0] = 0

            if sum(result_probas) == 0:
                result_word = method_word
            else:
                result_word = result_words[np.array(result_probas).argmax()]

        else:

            result_word = method_word
        return result_word

    def correct_words(self, text, topn=10, method='Lev_dict_FT', extend_dict=False):
        text = re.sub(r'[^\w\s]', '', text)
        # print(text)
        word_list = [word.lower() for word in text.split( ) if word.isalpha()]
        print(word_list)
        correct_words = []
        uncorrect_words = []
        correction_word = []
        k = 0
        for i, word in tqdm(enumerate(word_list)):
            try:
                checked_word = self.check_for_correction(word)
                if checked_word:
                    correct_words.append(checked_word)
                else:
                    k +=1
                    if not self.wordcount[self.wordcount['word'] == word].empty:
                        word_frequency = self.wordcount[self.wordcount['word'] == word]['counter'].values[0]
                    else:
                        word_frequency = 0

                    if extend_dict:
                        total_word_dict = list(set(self.total_word_dict + list(self.wordcount['word'].values)))

                    if method == 'FastText':
                        ft_words, ft_scores, models_distances = self.apply_fast_text(word, topn)
                        method_word = ft_words[0]

                    elif method == 'Levenshtein_dict':
                        lev_words = self.apply_levenshtein(word)
                        lev_word = lev_words[0]
                        method_word = lev_word

                    elif method == 'Levenshtein_top_ft':
                        ft_words, ft_scores, models_distances = self.apply_fast_text(word, topn)
                        lev_words = self.apply_levenshtein(word, ft_words)
                        lev_word = lev_words[0]
                        method_word = lev_word

                    elif method == 'Lev_dict_FT':
                        method_word = self.apply_levenshtein_and_model(word, topn)


                    uncorrect_words.append(str(word))
                    correction_word.append(str(method_word))
                    correct_words.append(str(method_word))
            except:
                print(word)
        # print()
        # print("Number of words corrected:" + str(k))
        # print("Corrected words:" + str(list(uncorrect_words)))
        # print()
        # return ' '.join(correct_words)
        # return list(uncorrect_words), list(correction_word), list(correct_words)
        return list(correct_words)