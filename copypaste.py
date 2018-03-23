import os
import re
import string
import math

DATA_DIR = 'DATA\\aclImdb\\train\\'
target_names = ['pos', 'neg']


def get_data(DATA_DIR):

    data = []
    target = []

    # spam
    spam_files = os.listdir(os.path.join(DATA_DIR, 'neg'))
    for spam_file in spam_files:
        with open(os.path.join(DATA_DIR, 'neg', spam_file), encoding="utf-8") as f:
            data.append(f.read())
            target.append(1)

    # ham
    ham_files = os.listdir(os.path.join(DATA_DIR, 'pos'))
    for ham_file in ham_files:
        with open(os.path.join(DATA_DIR, 'pos', ham_file), encoding="utf-8") as f:
            data.append(f.read())
            target.append(0)

    return data, target


class SpamDetector(object):
    """Implementation of Naive Bayes for binary classification"""

    def clean(self, s):
        # lager en translator som fjerner punctuation fra en tekststreng.
        # Returnerer den rensede strengen.
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

    def tokenize(self, text):
        # Tar en tekst (review), kaller clean-metoden, setter den til lower case.
        # Returnerer en liste med alle ordene i reviewen.
        text = self.clean(text).lower()
        return re.split("\W+", text)

    def get_word_counts(self, words):
        # Tar en string med ord, returnerer et vocabulary med hvert unike ord og antall ganger det finnes i strengen.
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts


    def fit(self, X, Y):
        # Oppretter dictionary for prior-sannsynlighet til klassene og antall ord og et vokabular-set.
        # Et set er en samling med unike elementer.
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()

        # n = antall reviews
        n = len(X)
        # Regner ut class priors for 'pos' og 'neg'.
        # Putter det i vocabularyet med klassen som key og sannsynligheten som value.
        self.log_class_priors['neg'] = math.log(sum(1 for label in Y if label == 1) / n)
        self.log_class_priors['pos'] = math.log(sum(1 for label in Y if label == 0) / n)

        # Oppretter to vocabularies inni word_counts: et for pos-ord og ett for neg-ord.
        # typ sånn her: { 'neg': { }, 'pos': { } }
        self.word_counts['neg'] = {}
        self.word_counts['pos'] = {}

        # Itererer gjennom både X og Y.
        # zip() gjør at den stopper å iterere når slutten av det korteste arrayet av X og Y er nådd.
        for x, y in zip(X, Y):
            # setter c til klassen av reviewen som itereres gjennom.
            c = 'neg' if y == 1 else 'pos'
            # Deler opp dokumentet i ord, fjerner puncutation,
            # og putter det inn i vokabularet counts med antall ganger hvert ord er i reviewen: {'ord' : x }.
            counts = self.get_word_counts(self.tokenize(x))

            # Itererer gjennom hvert item i counts.
            # Dvs: hvert unike ord i reviewen og valuen med antall ganger det finnes.
            for word, count in counts.items():
                # Hvis ordet ikke finnes i vocab (som skal ha alle unike ord) legges det inn i samlingen.
                if word not in self.vocab:
                    self.vocab.add(word)
                # Hvis ordet ikke finnes i det klasse-spesifikke vokabularet legges det inn der.
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0
                # Teller +1 i det klasse-spesifikke vokabularet for ordet.
                self.word_counts[c][word] += count


    def predict(self, X):
        result = []
        predicted = 0
        predicted100 = 0
        # Itererer gjennom alle reviews.
        for x in X:
            # Counts = alle de unike ordene i reviewen, med antall ganger hvert ord figurerer.
            counts = self.get_word_counts(self.tokenize(x))
            # Oppretter en score for begge klassene.
            spam_score = 0
            ham_score = 0
            # Itererer gjennom alle de unike ordene i reviewen.
            for word, _ in counts.items():
                # Gir en faen i om ordet ikke finnes i det globale vokabularet.
                if word not in self.vocab: continue

                # Regner ut log av antall ganger ordet finnes i det klassespesifikke vokabluaret,
                # deler på totalt antall ord i det klassespesifikke vokabularet antall unike ord totalt.
                # Legger til 1 for "laplance smoothing" i telleren ( i tilfelle telleren blir 0 ).
                log_w_given_spam = math.log((self.word_counts['neg'].get(word, 0.0) + 1) / (
                            sum(self.word_counts['neg'].values()) + len(self.vocab)))
                log_w_given_ham = math.log((self.word_counts['pos'].get(word, 0.0) + 1) / (
                            sum(self.word_counts['pos'].values()) + len(self.vocab)))
                # Legger sannsynligheten for ordet gitt klasse til i score.
                spam_score += log_w_given_spam
                ham_score += log_w_given_ham

            #Legger til sannsynligheten for at en review er av en klasse inn i score.
            spam_score += self.log_class_priors['neg']
            ham_score += self.log_class_priors['pos']

            # Legger til klassen med den høyeste scoren i result
            if spam_score > ham_score:
                result.append(1)
            else:
                result.append(0)
            predicted += 1
            if (predicted == (predicted100 + 100)):
                predicted100+=100
                print(str(predicted100))
        # returnerer en liste med result (1 eller 0) for alle reviews i X.
        return result

def main():
    testpath = 'DATA\\aclImdb\\test\\'
    # X = en array med hver review, y = en array med verdiene 1 eller 0.
    # Review og tall(klasse) kommer på samme plass i X og y.
    X, y = get_data(DATA_DIR)
    Xtest, ytest = get_data(testpath)
    # Oppretter et Spamdetector-object.
    # Kjører fit-metoden med reviews(X) og tilhørende klasser(y) som input.
    MNB = SpamDetector()
    MNB.fit(X, y)

    # Kjører predict-metoden på de 100 første reviews i X.
    # pred = den predictede klassen til de 100 første reviews i X.
    # true = Den reelle klassen til de 100 første reviews i X.
    pred = MNB.predict(Xtest[:100])
    true = ytest[:100]

    # Sjekker treffsikkerheten med å sammenlingne alle elementene(1 eller 0) i pred og true.
    accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))