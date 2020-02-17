# coding: utf-8

"""
 *
 * Copyright (C) 2018 Ciprian-Octavian Truică <ciprian.truica@cs.pub.ro>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
"""

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2020, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@cs.pub.ro"
__status__ = "Production"

class ChiSquared(object):
    def __init__(self, sentences, ws = 5):
        self.sentences = sentences
        self.ws = ws
        self.wordsSize = len([word for sentence in self.sentences for word in sentence])
        self.lenCollocations = 0.0
        self.chiDict = {}

    def build(self):
        leftW1 = {}
        rightW2 = {}
        for line in self.sentences:
            lineSize = len(line)
            for i in range(lineSize):
                w1 = line[i]

                if w1 not in self.chiDict:
                    self.chiDict[w1] = {}

                maxWindow = min(i + 1 + self.ws, lineSize)
                for j in range(i + 1,maxWindow):
                    w2 = line[j]

                    # store how many times w1 is on the right of a collocation
                    if w1 not in leftW1:
                        leftW1[w1] = 1.0
                    else:
                        leftW1[w1] += 1.0

                    # store how many times w2 is on the left of a collocation
                    if w2 not in rightW2:
                        rightW2[w2] = 1.0
                    else:
                        rightW2[w2] += 1.0

                    # store how many times w1 and w2 appear together
                    if w2 not in self.chiDict[w1]:
                        self.chiDict[w1][w2] = {'o11' : 1.0}
                    else:
                        self.chiDict[w1][w2]['o11'] += 1.0

                    # get the total collocations
                    self.lenCollocations += 1.0

        for w1 in self.chiDict:            
            for w2 in self.chiDict[w1]:
                # o11 is the number of times w1 and w2 create a collocation
                o11 = self.chiDict[w1][w2]['o11']
                # compute o12 as the number of time w2 
                # appears on the right of a collocation 
                # minus the number of times it is in a 
                # collocation with w1
                o12 = rightW2[w2] - o11

                # compute o21 as the number of time w1
                # appears on the left of a collocation 
                # minus the number of times it is in a 
                # collocation with w2
                o21 = leftW1[w1] - o11

                # compute o22 as the total number of collocations
                # minus the number of collocations where 
                # w1 and w2 appear
                o22 = float(self.lenCollocations) - o12 - o21 - o11

                # update dictionary
                self.chiDict[w1][w2]["o12"] = o12
                self.chiDict[w1][w2]["o21"] = o21                
                self.chiDict[w1][w2]["o22"] = o22
                numerator = float(self.wordsSize) * (o11 * o22 - o12 * o21) ** 2
                denominator = (o11 + o12) * (o11 + o21) * (o12 + o22) * (o21 + o22)
                if denominator == 0.0:
                    self.chiDict[w1][w2]["chisquared"] = 0.0
                else:
                    self.chiDict[w1][w2]["chisquared"] = numerator/denominator

    def getChiDict(self):
        return self.chiDict

    def getCollocsCandidates(self, w, order=True):
        return sorted(list(self.chiDict[w].items()), key=lambda x: x[1]["chisquared"], reverse=order)

    def saveToFile(self, filename):
        import ujson
        with open(filename, 'w') as f:
            ujson.dump(self.chiDict, f)

# this part is just for individual testing
if __name__ == "__main__":
    import sys
    import re
    import ujson
    import utils
    import static

    cachedStopWords_en = static.stopWordsEN()

    fileType = sys.argv[1]
    files = sys.argv[2:]
    if fileType == "txt":
        sentences, words, _ = utils.readTxtFiles(files)
    elif fileType == "csv":
        sentences, words, _ = utils.readCsvFiles(files)

    print("********************************")
    print("*          Chi-Squared         *")
    print("********************************")   
    cs = ChiSquared(sentences, 1)
    cs.build()
    chiDict = cs.getChiDict()

    # test get collocations
    # print(cs.getCollocsCandidates("kid"))
    # print(cs.chiDict['tum']['predilection'])
    # print(cs.chiDict['arthur']['ship'])

    # sort
    d = {}
    for w1 in chiDict:
        for w2 in chiDict[w1]:
            # do not take into account stopwords
            if w1 not in cachedStopWords_en and w2 not in cachedStopWords_en:
                d[str(w1) + " " + str(w2)] = chiDict[w1][w2]["chisquared"]
            
            
    # get top 20 
    s1 = [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]
    print(s1[0:20])
    
    # get bottom 20 
    s2 = [(k, d[k]) for k in sorted(d, key=d.get, reverse=False)]
    print(s2[0:20])

    # save
    with open('dict.json', 'w') as f:
         ujson.dump(chiDict, f)

