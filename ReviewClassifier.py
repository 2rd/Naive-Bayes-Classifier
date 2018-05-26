
import string
import math
from pathlib import Path
import pickle
import collections
import time

pos_path = Path('DATA/aclImdb/train/pos')
neg_path = Path('DATA/aclImdb/train/neg')
test_neg_path = Path('DATA/aclImdb/test/neg')
test_pos_path = Path('DATA/aclImdb/test/pos')
filetest = Path('10608_10.txt')
filetest2 = Path('7445_1.txt')

# Gjør innholdet i hver review om til en string og legger det i en array.
# Legger en review's tilhørende klasse på samme plass i target.
def get_reviews(pos_path, neg_path):
    print('\n' + '----------Retrieving reviews....--------------')
    start = time.time()
    reviews = []
    target = []

    for filename in pos_path.glob('**/*.txt'):
        reviews.append(filename.read_text(encoding='UTF-8'))
        target.append(1)

    for filename in neg_path.glob('**/*.txt'):
        reviews.append(filename.read_text(encoding='UTF-8'))
        target.append(0)
    end = time.time()
    print('Retrieving reviews took ' + '{0: .2f}'.format(end - start) + ' seconds to finish')
    return reviews, target

class reviewClassifier(object):

    def __init__(self, global_vocab, class_priors, prob_w_given_class, class_dictionaries):
        self.global_vocab = global_vocab
        self.class_priors = class_priors
        self.prob_w_given_class = prob_w_given_class
        self.class_dictionaries = class_dictionaries

    def fit(self, X, Y):
        print('\n' + '-------- Please wait while the model is being trained....-----------')
        start = time.time()
        # n = antall reviews
        total_reviews = len(X)

        # teller antall positive og negative filer og regner ut log prior probabilty til klassene.
        total_posfiles = sum(1 for c in Y if c == 1)
        total_negfiles = sum(1 for c in Y if c == 0)

        self.class_priors['pos'] = math.log(total_posfiles/total_reviews)
        self.class_priors['neg'] = math.log(total_negfiles/total_reviews)

        # Oppretter et dictionary for hver av klassene i dictionary.
        self.class_dictionaries['pos'] = {}
        self.class_dictionaries['neg'] = {}

        # Itererer gjennom både X og Y.
        # zip() gjør at den stopper å iterere når slutten av det korteste arrayet av X og Y er nådd.
        for x, y in zip(X, Y):

            # setter c til klassen av reviewen som itereres gjennom.
            c = 'pos' if y == 1 else 'neg'

            # Tar hvert ord fra dokumentet og lager et dictionary m/hjelp av Counter
            vocab = collections.Counter()
            vocab.update(self.text_to_words(x))

            # Tar hvert unike ord i reviewen og oppdaterer valuen med antall ganger det finnes.
            # Putter hvert unike ord i det globale vokabularet.
            for word, count in vocab.items():
                if word in self.class_dictionaries[c]:
                    self.class_dictionaries[c][word] += 1
                else:
                    self.class_dictionaries[c][word] = 1
                if word in self.global_vocab:
                    self.global_vocab.add(word)

        self.prob_w_given_class['pos'] = {}
        self.prob_w_given_class['neg'] = {}

        # Regner ut sannsynligheten for hvert ord gitt klasse
        for word in self.class_dictionaries['pos']:
            self.prob_w_given_class['pos'][word] = math.log((self.class_dictionaries['pos'].get(word, 0.0) + 1) /
                                                            (sum(self.class_dictionaries['pos'].values()) + len(self.global_vocab)))
        for word in self.class_dictionaries['neg']:
            self.prob_w_given_class['neg'][word] = math.log((self.class_dictionaries['neg'].get(word, 0.0) + 1) /
                                                            (sum(self.class_dictionaries['neg'].values()) + len(self.global_vocab)))

        end = time.time()
        print('Training took ' + '{0: .3f}'.format(end - start) + ' seconds to finish')

    def p_w_given_c(self, c, word):
        result = math.log((self.class_dictionaries[c].get(word, 0.0) + 1) /
                                                        (sum(self.class_dictionaries[c].values()) + len(
                                                            self.global_vocab)))
        return result

    #Lager en prediction på hver review i X på om den er positiv eller negativ.
    def predict(self, X):
        print('\n' + "Categorizing reviews... ")
        start = time.time()
        result = []

        for x in X:
            vocab = collections.Counter()
            vocab.update(self.text_to_words(x))

            pos_score = 1
            neg_score = 1

            for word in vocab:

                pos_score += self.p_w_given_c('pos', word)
                neg_score += self.p_w_given_c('neg', word)

            pos_score += self.class_priors['pos']
            neg_score += self.class_priors['neg']

            if(pos_score > neg_score):
                result.append(1)
            else:
                result.append(0)
        end = time.time()
        print('Categorizing reviews took ' + '{0: .2f}'.format(end - start) + ' seconds to finish')
        return result

    # Tar en reviewtekst, fjerner punctuation og returnerer en array med alle ordene den inneholder.
    def text_to_words(self, review_text):
        translator = str.maketrans("", "", string.punctuation)
        review_text = review_text.lower()
        review_text = review_text.translate(translator)

        return review_text.split()

    #takes a review and categorizes it, either positive or negative
    def categorize(self, filepath):
        review = Path(filepath)
        result = self.predict([review.read_text(encoding='UTF-8 ')])
        for i in result:
            if i == 1:
                print('positive')
            else:
                print('negative')

# Tester modellen i accuracy, precision, recall, og f-measure.
# paramerter 'no_of_reviews' : (integer) antall reviews modellen skal testes på
def test(no_of_reviews):
    test_X, test_y = get_reviews(test_pos_path, test_neg_path)

    # prediction = de predictede klassene til testreviews.
    # correct = De reelle klassene tilhørende reviews.
    prediction = model.predict(test_X[:no_of_reviews])
    correct = test_y[:no_of_reviews]

    tp = sum(1 for i in range(len(prediction)) if ((prediction[i] == 1) and (correct[i] == 1)))
    fp = sum(1 for i in range(len(prediction)) if ((prediction[i] == 1) and (correct[i] == 0)))
    tn = sum(1 for i in range(len(prediction)) if ((prediction[i] == 0) and (correct[i] == 0)))
    fn = sum(1 for i in range(len(prediction)) if ((prediction[i] == 0) and (correct[i] == 1)))

    # Sjekker treffsikkerheten med å sammenlingne alle elementene(1 eller 0) i predicition og correct.
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f = 2*((precision*recall)/(precision+recall))

    print("\n" + "Number of reviews tested: " + str(no_of_reviews))
    print("accuracy:  " + "{0:.3f}".format(accuracy))
    print("precision:  " + "{0:.3f}".format(precision))
    print("recall:    "+ "{0:.3f}".format(recall))
    print("f-measure:     "+ "{0:.3f}".format(f))

#Laster modellen, eller lager ny modell, dersom den ikke eksisterer.
try:
    model = pickle.load(open('model.pickle', 'rb'))
except(OSError, IOError) as e:
    # train_X = en array med hver review, train_y = en array med verdiene 1 eller 0.
    # Review og tall(klasse) kommer på samme plass i train_X og train_y.
    train_X, train_y = get_reviews(pos_path, neg_path)
    # Oppretter et reviewClassifier-objekt
    model = reviewClassifier(set(), {}, {}, {})
    # Kjører fit-metoden med reviews(train_X) og tilhørende klasser(train_y) som input.
    model.fit(train_X, train_y)

# Lagre modellen
pickle.dump(model, open('model.pickle', 'wb'))
