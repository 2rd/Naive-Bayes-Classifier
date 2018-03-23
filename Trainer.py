
# Trenger sannsynlighetene for:
#   - Klasse(pos eller neg)
#   - Ord
#   - Ord gitt klasse

import os
import string
import math


posPath = 'DATA\\aclImdb\\train\\pos'
negPath = 'DATA\\aclImdb\\train\\neg'

def main():



    posPath = 'C:\\Users\\tordk\\Documents\\INFO284\\1rstGroupAssingment\\300_pos'
    negPath = 'C:\\Users\\tordk\\Documents\\INFO284\\1rstGroupAssingment\\300_neg'

    posWords = wordParser(posPath)
    negWords = wordParser(negPath)

    posVocab = vocabulary(posWords)
    negVocab = vocabulary(negWords)

    numOfPosFiles = len([file for file in os.listdir(posPath)])
    numOfNegFiles = len([file for file in os.listdir(negPath)])
    totalFiles = numOfPosFiles + numOfNegFiles
    probOfClass1 = math.log10(numOfPosFiles/totalFiles)
    probOfClass0 = math.log10(numOfNegFiles/totalFiles)
    print(str(probOfClass1))
    print(str(probOfClass0))

    posWordProbability = wordProbability(posVocab, posWords.__len__())
    negWordProbability = wordProbability(negVocab, negWords.__len__())

    print('antall ganger bad i positiv: ' + str(posVocab.get('bad')))
    print('sannsynlighet for bad i positiv: ' + str(posWordProbability.get('bad')))
    print('antall ganger bad i negativ: ' + str(negVocab.get('bad')))
    print('sannsynlighet for bad i negativ: ' + str(negWordProbability.get('bad')))

# Parser alle dokumentene i en mappe og returnerer en liste med alle ordene de inneholder.
def wordParser(path):


    a = list()
    translator = str.maketrans('', '', string.punctuation)
    # path = 'C:\\Users\\tordk\\Documents\\INFO284\\1rstGroupAssingment\\300_pos'
    for filename in os.listdir(path):
        with open(path + '\\' + filename, encoding='utf-8') as f:

            for line in f:
                for word in line.split():
                    word = word.lower()
                    word = word.translate(translator)

                    a.append(word)
    return a

# Tar en liste med ord og returnerer en dictionary med hvert unike ord og antall ganger de står i listen.
def vocabulary(words):
    vocab = dict()

    for word in words:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1
    return vocab



# Returnerer sannsynligheten av noe... nk
def wordProbability(vocabulary, totalWords):
    newVocab = dict()
    for word in vocabulary:

        newVocab[word] = vocabulary.get(word)/totalWords

    return newVocab
