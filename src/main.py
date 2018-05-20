# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2017, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@cs.pub.ro"
__status__ = "Production"


from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import csr_matrix
from multiprocessing import cpu_count
from ddl import MongoDBConnector
from ate import AutomaticTermExtraction
from tm import TopicModeling
import evaluation_measures
import numpy as np
import sys
import math
from time import time
from collections import Counter


# parametters
grammar ={
         "P1":  "P1: {<NN.*>}",
         "P2":  "P2: {<NN.*> (<IN|RP|TO>)? <NN.*>}",
         "P3":  "P3: {<JJ.*> (<IN|RP|TO>)? <NN.*>}",
         "P4":  "P4: {<NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
         "P5":  "P5: {<JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
         "P6":  "P6: {<NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
         "P7":  "P7: {<JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
         "P8":  "P8: {<NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
         "P9":  "P9: {<JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P10": "P10: {<NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P11": "P11: {<NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
        "P12": "P12: {<JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P13": "P13: {<JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
        "P14": "P14: {<NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
        "P15": "P15: {<NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P16": "P16: {<JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P17": "P17: {<NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P18": "P18: {<NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P19": "P19: {<NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
        "P20": "P20: {<JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P21": "P21: {<JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P22": "P22: {<JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
        "P23": "P23: {<NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P24": "P24: {<NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
        "P25": "P25: {<JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <NN.*>}",
        "P26": "P26: {<JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",
        "P27": "P27: {<JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <JJ.*> (<IN|RP|TO>)? <NN.*>}",

}

# punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~-'
# punctuation = '!"#$%&\'()*,./:;<=>?@[\\]^_`{|}~'
punctuation = '"#$%&\'()*+/<=>@[\\]^_`{|}~-'

threshold = 1.0

docs = []
doc2class = {}

word2id = {}
id2word = {}

doc2id = {}
id2doc = {}

wordIDF = {}

topicDocs = []

tmLabels = []
topicNGrams = {}
topicNGramsList = []


no_threads = cpu_count()
tags = set()
i = 0

treshold = 1.0
min_ngram_len = 2
candidateNGramsFreq = []

def processElement(elem):
    # print('process elem')
    text = elem['cleanText']
    lt = AutomaticTermExtraction(text, grammar, punctuation)
    lt.getLematizedSentences()
    lt.getNGrams()
    lt.computeCValue(treshold=threshold)
    doc = {}
    doc['text'] = text
    doc['docID'] = elem['_id']
    doc['tag'] = elem['tags'][0]
    doc['terms'] = lt.getVocabulary()
    doc['candidateNGrams'] = lt.getCandidateNGrams()
    doc['allNGrams'] = lt.getAllNGrams()
    l_tf = {}
    l_count = {}
    l_idf = {}
    lenDoc = 0
    for word in elem['words']:
        if word['word'] in doc['terms']:
            l_tf[word['word']] = word['tf']
            l_count[word['word']] = word['count']
            lenDoc += word['count']
            l_idf[word['word']] = wordIDF[word['word']]
    doc['wordsTF'] = l_tf
    doc['wordsCount'] = l_count
    doc['lenDoc'] = lenDoc
    doc['wordsIDF'] = l_idf
    return doc

def processVocabulary(elem):
    return {elem['word']: elem['IDF']}

# build tfidf & okapi matrix
def buildMatrix(documents, k1=1.6, b=0.75):
    idx_doc = 0
    idx_word = 0
    data_tfidf = []
    data_okapi = []
    row = []
    col = []
    avgDL = sum([doc['lenDoc'] for doc in documents])/float(len(documents))
    for doc in documents:
        # print('process docuemnt', idx_doc)
        tags.add(doc['tag'])
        doc2id[doc['docID']] = idx_doc
        id2doc[idx_doc] = doc['docID']
        doc2class[doc2id[doc['docID']]] = doc['tag']
        for word in doc['wordsTF']:
            if not word2id.get(word):
                word2id[word] = idx_word
                id2word[idx_word] = word
                idx_word += 1
            tf = doc['wordsTF'][word]
            idf = doc['wordsIDF'][word]
            tfidf = tf * idf
            docLen = doc['lenDoc']
            okapi = (tf * idf * (k1 + 1))/(tf + k1*(1-b+b*(docLen/avgDL)))
            row.append(doc2id[doc['docID']])
            col.append(word2id[word])
            data_tfidf.append(tfidf)
            data_okapi.append(okapi)
        idx_doc += 1

    csr_tfidf = csr_matrix((data_tfidf, (row, col)), dtype=np.float64)
    csr_okapi = csr_matrix((data_okapi, (row, col)), dtype=np.float64)
    return csr_tfidf, csr_okapi

def getNGramsTopic(elem):
    for td in topicDocs:
        if doc2id[elem['docID']] in td['docs']:
            return {"topic_id": td["topic_id"], "ngram": elem['candidateNGrams'], 'allngrams': elem['allNGrams']}

def computeLabelsCValue(ngram):
    if len(ngram) < min_ngram_len:
        return None
    s = 0
    c = 0
    for nested_ngram in candidateNGramsFreq:
        if set(ngram).issubset(set(nested_ngram)) and ngram != nested_ngram: 
            c += 1
            s += candidateNGramsFreq[nested_ngram]
            for t_ngram in candidateNGramsFreq:
                if set(nested_ngram).issubset(set(t_ngram)) and t_ngram != nested_ngram and ngram != t_ngram and set(ngram).issubset(set(t_ngram)):
                    s -= candidateNGramsFreq[t_ngram]
    if c > 0:
        cvalue = math.log(1 + len(ngram), 2) * (candidateNGramsFreq[ngram] - s/c)
    else:
        cvalue = math.log(1 + len(ngram), 2) * candidateNGramsFreq[ngram]
    if cvalue < threshold:
        return None
    return { ngram:  cvalue }

# params:
# dbname Database
# num_iter number of iterations
if __name__ == "__main__":
    dbname = sys.argv[1]
    num_iter = int(sys.argv[2])
    no_tests = int(sys.argv[3])
    print("Connecting to database...")
    conn = MongoDBConnector(dbname=dbname)
    print("Connected!")
    # get all the words from the vocabulary

    print("Vocabulary...")
    voc = conn.getRecords(collection='vocabulary', projection={'word':1, 'IDF': 1})

    with ProcessPoolExecutor(max_workers=no_threads) as worker:
        for result in worker.map(processVocabulary, voc):
            if result:
                wordIDF.update(result)
    print("Vocabulary processed!")
    # get all the documents
    # filter={"_id": {"$in" : ["2", "5", "4", "3", "6", "7", "9", "8", "1", "13", "463", "465", "464", "466", "467", "468", "470", "469", "472", "473"]}},
    print("Documents...")
    documents = conn.getRecords(collection='documents', projection={'cleanText': 1, 'tags': 1, 'words': 1})
    
    with ProcessPoolExecutor(max_workers=no_threads) as worker:
        for result in worker.map(processElement, documents):
            if result:
                docs.append(result)
    conn.closeConection()

    print("Documents processed!")
    print(len(docs))

    print("Vectorisation started...")
    csr_tfidf, csr_okapi = buildMatrix(documents=docs)
    print("Vectorisation processed!")

    num_topics = len(tags)
    print('Num topics', num_topics)
    print('len voc small:', len(id2word))
    
    print("\n\n=========================================================")
    print("==========================TFIDF==========================")
    print("=========================================================\n\n")
    for i in range(0, no_tests):
        print("TFIDF")
        print('NMF TFIDF with cvalue:')
        
        topicNGrams = {}
        topicAllNGrams = {}
        topicNGramsList = []
        tmLabels = []

        topic_model = TopicModeling(id2word=id2word, corpus=csr_tfidf, doc2class=doc2class, num_cores=30)

        startTime = time()
        print("Starting the TM process...")
        
        topics = topic_model.topicsNMF(num_topics=num_topics, num_iterations=num_iter)
        for topic in topics:
            wTopics = []
            for words in topic[1]:
                wTopics.append(words[0])
            print("Topic", topic[0], wTopics)
            topicNGrams[topic[0]] = []
            topicAllNGrams[topic[0]] = {}

        intermediateTime1 = time()
        print("NMF TFIDF C-Value TM Time:", (intermediateTime1 - startTime))
        print("Starting the ngram process...")

        
        # get the candidate ngrams for each topic
        topicDocs = topic_model.topicDocsNMF
        with ProcessPoolExecutor(max_workers=no_threads) as worker:
            for result in worker.map(getNGramsTopic, docs):
                if result:
                    topicNGrams[result["topic_id"]] += result['ngram']
                    for n in result['allngrams']:
                        if topicAllNGrams[result["topic_id"]].get(n):
                            for ngram in result['allngrams'][n]:
                                if topicAllNGrams[result["topic_id"]][n].get(ngram):
                                    topicAllNGrams[result["topic_id"]][n][ngram] += result['allngrams'][n][ngram]
                                else:
                                    topicAllNGrams[result["topic_id"]][n][ngram] = result['allngrams'][n][ngram]
                        else:
                            topicAllNGrams[result["topic_id"]][n] = result['allngrams'][n]

        intermediateTime2 = time()
        print("NMF TFIDF C-Value N-Grams Time:", (intermediateTime2 - intermediateTime1))
        print("Starting the labling process...")

        # get the lable for each topic
        for topic_id in topicNGrams:
            it0 = time()
            print("Starting labeling topic:", topic_id)
            candidateNGramsFreq = Counter(topicNGrams[topic_id])
            weightedLebels = {}
            with ProcessPoolExecutor(max_workers=no_threads) as worker:
                for result in worker.map(computeLabelsCValue, candidateNGramsFreq):
                    if result:
                        weightedLebels.update(result)
            label = sorted(weightedLebels, key = weightedLebels.get, reverse=True)[0]
            tmLabels.append({"topic": topic_id, "label": label})
            it1 = time()
            print("Finished labeling topic:", topic_id, "with", label, "in", (it1 - it0))
            pmi, npmi = evaluation_measures.pmiBigram(topicAllNGrams[topic_id], label)
            print("pmi:", pmi, "npmi:", npmi)

        for topiclabel in tmLabels:
            print(topiclabel)

        endTime = time()
        print("NMF TFIDF C-Value Labels Time:", (endTime - intermediateTime2))
        print("NMF TFIDF C-Value Total Time:", (endTime - startTime))
        print('NMF TFIDF ARI C-Value:', evaluation_measures.adj_rand_index(topic_model.doc2topicNMF))

        print("\n*********************************************************\n")

        print('LDA TFIDF with cvalue:')

        topicNGrams = {}
        topicAllNGrams = {}
        topicNGramsList = []
        tmLabels = []

        startTime = time()
        print("Starting the TM process...")

        topic_model = TopicModeling(id2word=id2word, corpus=csr_tfidf, doc2class=doc2class, num_cores=30)
        topics = topic_model.topicsLDA(num_topics=num_topics, num_iterations=num_iter)
        for topic in topics:
            wTopics = []
            for words in topic[1]:
                wTopics.append(words[0])
            print("Topic", topic[0], wTopics)
            topicNGrams[topic[0]] = []
            topicAllNGrams[topic[0]] = {}

        intermediateTime1 = time()
        print("LDA TFIDF C-Value TM Time:", (intermediateTime1 - startTime))
        print("Starting the ngram process...")

        topicDocs = topic_model.topicDocsLDA
        with ProcessPoolExecutor(max_workers=no_threads) as worker:
            for result in worker.map(getNGramsTopic, docs):
                if result:
                    topicNGrams[result["topic_id"]] += result['ngram']
                    for n in result['allngrams']:
                        if topicAllNGrams[result["topic_id"]].get(n):
                            for ngram in result['allngrams'][n]:
                                if topicAllNGrams[result["topic_id"]][n].get(ngram):
                                    topicAllNGrams[result["topic_id"]][n][ngram] += result['allngrams'][n][ngram]
                                else:
                                    topicAllNGrams[result["topic_id"]][n][ngram] = result['allngrams'][n][ngram]
                        else:
                            topicAllNGrams[result["topic_id"]][n] = result['allngrams'][n]

        intermediateTime2 = time()
        print("LDA TFIDF C-Value N-Grams Time:", (intermediateTime2 - intermediateTime1))
        print("Starting the labling process...")

        for topic_id in topicNGrams:
            it0 = time()
            print("Starting labeling for topic:", topic_id)
            candidateNGramsFreq = Counter(topicNGrams[topic_id])
            weightedLebels = {}
            with ProcessPoolExecutor(max_workers=no_threads) as worker:
                for result in worker.map(computeLabelsCValue, candidateNGramsFreq):
                    if result:
                        weightedLebels.update(result)
            label = sorted(weightedLebels, key = weightedLebels.get, reverse=True)[0]
            tmLabels.append({"topic": topic_id, "label": label})
            it1 = time()
            print("Finished labeling topic:", topic_id, "with", label, "in", (it1 - it0))
            pmi, npmi = evaluation_measures.pmiBigram(topicAllNGrams[topic_id], label)
            print("pmi:", pmi, "npmi:", npmi)

        for topiclabel in tmLabels:
            print(topiclabel)

        endTime = time()
        print("LDA TFIDF C-Value Labels Time:", (endTime - intermediateTime2))
        print("LDA TFIDF C-Value Total Time:", (endTime - startTime))
        print('LDA TFIDF ARI c-value:', evaluation_measures.adj_rand_index(topic_model.doc2topicLDA))


    print("\n\n=========================================================")
    print("=======================Okapi BM25========================")
    print("=========================================================\n\n")
    for i in range(0, no_tests):
        print('NMF Okapi BM25 with cvalue:')

        topicNGrams = {}
        topicAllNGrams = {}
        topicNGramsList = []
        tmLabels = []

        topic_model = TopicModeling(id2word=id2word, corpus=csr_okapi, doc2class=doc2class, num_cores=30)

        startTime = time()
        print("Starting the TM process...")

        topics = topic_model.topicsNMF(num_topics=num_topics, num_iterations=num_iter)
        for topic in topics:
            wTopics = []
            for words in topic[1]:
                wTopics.append(words[0])
            print("Topic", topic[0], wTopics)
            topicNGrams[topic[0]] = []
            topicAllNGrams[topic[0]] = {}

        intermediateTime1 = time()
        print("NMF Okapi BM25 C-Value TM Time:", (intermediateTime1 - startTime))
        print("Starting the ngram process...")

        topicDocs = topic_model.topicDocsNMF
        with ProcessPoolExecutor(max_workers=no_threads) as worker:
            for result in worker.map(getNGramsTopic, docs):
                if result:
                    topicNGrams[result["topic_id"]] += result['ngram']
                    for n in result['allngrams']:
                        if topicAllNGrams[result["topic_id"]].get(n):
                            for ngram in result['allngrams'][n]:
                                if topicAllNGrams[result["topic_id"]][n].get(ngram):
                                    topicAllNGrams[result["topic_id"]][n][ngram] += result['allngrams'][n][ngram]
                                else:
                                    topicAllNGrams[result["topic_id"]][n][ngram] = result['allngrams'][n][ngram]
                        else:
                            topicAllNGrams[result["topic_id"]][n] = result['allngrams'][n]

        intermediateTime2 = time()
        print("NMF Okapi BM25 C-Value N-Grams Time:", (intermediateTime2 - intermediateTime1))
        print("Starting the labling process...")

       # get the lable for each topic
        for topic_id in topicNGrams:
            it0 = time()
            print("Starting labeling for topic:", topic_id)
            candidateNGramsFreq = Counter(topicNGrams[topic_id])
            weightedLebels = {}
            with ProcessPoolExecutor(max_workers=no_threads) as worker:
                for result in worker.map(computeLabelsCValue, candidateNGramsFreq):
                    if result:
                        weightedLebels.update(result)
            label = sorted(weightedLebels, key = weightedLebels.get, reverse=True)[0]
            tmLabels.append({"topic": topic_id, "label": label})
            it1 = time()
            print("Finished labeling topic:", topic_id, "with", label, "in", (it1 - it0))
            pmi, npmi = evaluation_measures.pmiBigram(topicAllNGrams[topic_id], label)
            print("pmi:", pmi, "npmi:", npmi)

        for topiclabel in tmLabels:
            print(topiclabel)

        endTime = time()
        print("NMF Okapi BM25 C-Value Labels Time:", (endTime - intermediateTime2))
        print("NMF Okapi BM25 C-Value Total Time:", (endTime - startTime))
        print('NMF Okapi BM25 ARI c-value:', evaluation_measures.adj_rand_index(topic_model.doc2topicNMF))

        print("\n*********************************************************\n")

        print('LDA Okapi BM25 with cvalue:')

        topicNGrams = {}
        topicAllNGrams = {}
        topicNGramsList = []
        tmLabels = []

        startTime = time()
        print("Starting the TM process...")

        topic_model = TopicModeling(id2word=id2word, corpus=csr_okapi, doc2class=doc2class, num_cores=30)
        topics = topic_model.topicsLDA(num_topics=num_topics, num_iterations=num_iter)
        for topic in topics:
            wTopics = []
            for words in topic[1]:
                wTopics.append(words[0])
            print("Topic", topic[0], wTopics)
            topicNGrams[topic[0]] = []
            topicAllNGrams[topic[0]] = {}

        intermediateTime1 = time()
        print("LDA Okapi BM25 C-Value TM Time:", (intermediateTime1 - startTime))
        print("Starting the ngram process...")

        topicDocs = topic_model.topicDocsLDA
        with ProcessPoolExecutor(max_workers=no_threads) as worker:
            for result in worker.map(getNGramsTopic, docs):
                if result:
                    topicNGrams[result["topic_id"]] += result['ngram']
                    for n in result['allngrams']:
                        if topicAllNGrams[result["topic_id"]].get(n):
                            for ngram in result['allngrams'][n]:
                                if topicAllNGrams[result["topic_id"]][n].get(ngram):
                                    topicAllNGrams[result["topic_id"]][n][ngram] += result['allngrams'][n][ngram]
                                else:
                                    topicAllNGrams[result["topic_id"]][n][ngram] = result['allngrams'][n][ngram]
                        else:
                            topicAllNGrams[result["topic_id"]][n] = result['allngrams'][n]

        intermediateTime2 = time()
        print("LDA Okapi BM25 C-Value N-Grams Time:", (intermediateTime2 - intermediateTime1))
        print("Starting the labeling process...")

        # get the lable for each topic
        for topic_id in topicNGrams:
            it0 = time()
            print("Starting labeling for topic:", topic_id)
            candidateNGramsFreq = Counter(topicNGrams[topic_id])
            weightedLebels = {}
            with ProcessPoolExecutor(max_workers=no_threads) as worker:
                for result in worker.map(computeLabelsCValue, candidateNGramsFreq):
                    if result:
                        weightedLebels.update(result)
            label = sorted(weightedLebels, key = weightedLebels.get, reverse=True)[0]
            tmLabels.append({"topic": topic_id, "label": label})
            it1 = time()
            print("Finished labeling topic:", topic_id, "with", label, "in", (it1 - it0))
            pmi, npmi = evaluation_measures.pmiBigram(topicAllNGrams[topic_id], label)
            print("pmi:", pmi, "npmi:", npmi)

        for topiclabel in tmLabels:
            print(topiclabel)


        endTime = time()
        print("LDA Okapi BM25 C-Value Labels Time:", (endTime - intermediateTime2))
        print("LDA Okapi BM25 C-Value Total Time:", (endTime - startTime))
        print('LDA Okapi ARI c-value:', evaluation_measures.adj_rand_index(topic_model.doc2topicLDA))
