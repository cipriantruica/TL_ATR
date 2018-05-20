# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2017, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@cs.pub.ro"
__status__ = "Production"


from nltk import word_tokenize
from nltk import pos_tag
from nltk import sent_tokenize
from nltk import RegexpParser
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words
from nltk.corpus import stopwords
from collections import Counter
import math

class AutomaticTermExtraction:
    def __init__(self, text, grammar, punctuation):
        self.text = text.lower()        
        self.grammar = grammar
        self.punctuation = punctuation
        self.wnl = WordNetLemmatizer()
        self.sentences = []
        self.sentencesTokens = []
        self.candidateNGrams = []
        self.candidateNGramsFreq = {}
        self.cvalue = {}
        self.vocabulary = set([])
        self.stopwords = self.stopWordsEN()
        self.allNGrams = {}
    
    def stopWordsEN(self):
        sw_stop_words = get_stop_words('en')
        sw_nltk = stopwords.words('english')
        sw_mallet = ['a', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', 'different', 'do', 'does', 'doing', 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'happens', 'hardly', 'has', 'have', 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', 'way', 'we', 'welcome', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', 'wonder', 'would', 'would', 'x', 'y', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']
        return list(set(sw_stop_words + sw_nltk + sw_mallet))

    def removePunctuationText(self, text):
        for c in self.punctuation:
            text = text.replace(c, ' ')
        return text

    def getLematizedSentences(self):
        for sentence in sent_tokenize(self.text):
            lemma_sentence = []
            tokens = []
            for word_pos in pos_tag(word_tokenize(self.removePunctuationText(sentence))):
                pos = word_pos[1][0].lower().replace('j', 'a')
                word = word_pos[0]
                if pos in ['n', 'a', 'v', 'r']:
                    word = self.wnl.lemmatize(word_pos[0], pos)
                lemma_sentence.append((word, word_pos[1]))
                tokens.append(word)

            # print(lemma_sentence)
            if len(lemma_sentence) > 0:
                self.sentences.append(lemma_sentence)
            if len(tokens) > 0:
                self.sentencesTokens.append(tokens)


    def getNGrams(self):
        for chunk_type in self.grammar:
            self.cp = RegexpParser(self.grammar[chunk_type])
            for sentences in self.sentences:
                chunked_sents = self.cp.parse(sentences)
                # print(chunked_sents)
                for subtree in chunked_sents.subtrees():
                    if subtree.label() == chunk_type:
                        self.candidateNGrams.append(tuple(word[0] for word in subtree))
        self.candidateNGramsFreq = Counter(self.candidateNGrams)
        # for elem in self.candidateNGramsFreq:
        #     print(elem, self.candidateNGramsFreq[elem])
        # print('\n\n')

    def computeCValue(self, treshold=0.0, candidateNGramsFreq=None):
        if candidateNGramsFreq is not None:
            self.candidateNGramsFreq = candidateNGramsFreq
        for ngram in self.candidateNGramsFreq:
            s = 0
            c = 0
            # print('candidate:', ngram, self.candidateNGramsFreq[ngram])
            for nested_ngram in self.candidateNGramsFreq:
                if set(ngram).issubset(set(nested_ngram)) and ngram != nested_ngram: 
                    # print('nested:', nested_ngram, self.candidateNGramsFreq[nested_ngram])
                    c += 1
                    s += self.candidateNGramsFreq[nested_ngram]
                    for t_ngram in self.candidateNGramsFreq:
                        if set(nested_ngram).issubset(set(t_ngram)) and t_ngram != nested_ngram and ngram != t_ngram and set(ngram).issubset(set(t_ngram)):
                            # print('snc nested', t_ngram, self.candidateNGramsFreq[t_ngram])
                            s -= self.candidateNGramsFreq[t_ngram]

            # print(ngram, c, s)
            if c > 0:
                self.cvalue[ngram] = math.log(1 + len(ngram), 2) * (self.candidateNGramsFreq[ngram] - s/c)
            else:
                self.cvalue[ngram] = math.log(1 + len(ngram), 2) * self.candidateNGramsFreq[ngram]

        # print(sorted(self.cvalue, key = self.cvalue.get))
        
        for elem in sorted(self.cvalue, key = self.cvalue.get, reverse=True):
            if self.cvalue[elem] >= treshold:
                for word in elem:
                    if word not in self.stopwords:                        
                        self.vocabulary.add(word)
        
    def getVocabulary(self):
        # print(self.vocabulary)
        return list(self.vocabulary)

    def getCValue(self):
        return self.cvalue

    def getCandidateNGrams(self):
        return self.candidateNGrams

    def getAllNGrams(self):
        maxN = 2
        for ngram in self.cvalue:
            if maxN < len(ngram):
                maxN = len(ngram)

        for n in range(1, maxN + 1):
            nGramDic = {}
            for sentence in self.sentencesTokens:
                for ngram in ngrams(sentence, n):
                    if nGramDic.get(ngram):
                        nGramDic[ngram] += 1
                    else:
                        nGramDic[ngram] = 1
            self.allNGrams[n] = nGramDic

        return self.allNGrams

