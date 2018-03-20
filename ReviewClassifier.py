import os
import string
import math

pos_path = 'DATA\\aclImdb\\train\\pos'
neg_path = 'DATA\\aclImdb\\train\\neg'
test_neg_path = 'DATA\\aclImdb\\test\\neg'
test_pos_path = 'DATA\\aclImdb\\test\\pos'

target_names = ['pos', 'neg']

def main(no_of_testreviews):

    # X = en array med hver review, y = en array med verdiene 1 eller 0.
    # Review og tall(klasse) kommer på samme plass i X og y.
    X, y = get_reviews(pos_path, neg_path)

    # Oppretter et reviewClassifier-objekt
    # Kjører fit-metoden med reviews(X) og tilhørende klasser(y) som input.
    MNB = reviewClassifier()
    MNB.fit(X, y)

    test_X, test_y = get_reviews(test_pos_path, test_neg_path)

    # prediction = de predictede klassene til testreviews.
    # correct = De reelle klassene tilhørende reviews.
    prediction = MNB.predict(test_X[:no_of_testreviews])
    correct = test_y[:no_of_testreviews]

    # Sjekker treffsikkerheten med å sammenlingne alle elementene(1 eller 0) i predicition og correct.
    accuracy = sum(1 for i in range(len(prediction)) if prediction[i] == correct[i]) / float(len(prediction))
    print("{0:.3f}".format(accuracy))

# Gjør innholdet i hver review om til en string og legger det i en array.
# Legger en review's tilhørende klasse på samme plass i target.
def get_reviews(pos_path, neg_path):
    reviews = []
    target = []

    for filename in os.listdir(pos_path):
        with open(pos_path + '\\' + filename, encoding='utf-8') as f:
            reviews.append(f.read())
            target.append(1)

    for filename in os.listdir(neg_path):
        with open(neg_path + '\\' + filename, encoding='utf-8') as f:
            reviews.append(f.read())
            target.append(0)

    return reviews, target

class reviewClassifier(object):

    global_vocab = set()
    class_priors = {}
    prob_w_given_class = {}

    def fit(self, X, Y):
        class_dictionaries = {}
        # n = antall reviews
        total_reviews = len(X)

        # teller antall positive og negative filer og regner ut log prior probabilty til klassene.
        total_posfiles = sum(1 for c in Y if c == 1)
        total_negfiles = sum(1 for c in Y if c == 0)
        self.class_priors['pos'] = math.log(total_posfiles/total_reviews)
        self.class_priors['neg'] = math.log(total_negfiles/total_reviews)

        # Oppretter et dictionary for hver av klassene i dictionary.
        class_dictionaries['pos'] = {}
        class_dictionaries['neg'] = {}

        # Itererer gjennom både X og Y.
        # zip() gjør at den stopper å iterere når slutten av det korteste arrayet av X og Y er nådd.
        for x, y in zip(X, Y):

            # setter c til klassen av reviewen som itereres gjennom.
            c = 'pos' if y == 1 else 'neg'

            # Deler opp dokumentet i ord, fjerner puncutation,
            # og putter det inn i dictionaries med antall ganger hvert ord er i reviewen.
            vocab = self.word_count(self.text_to_words(x))

            # Tar hvert unike ord i reviewen og oppdaterer valuen med antall ganger det finnes.
            # Putter hvert unike ord i det globale vokabularet.
            for word, count in vocab.items():
                if word in class_dictionaries[c]:
                    class_dictionaries[c][word] += 1
                else:
                    class_dictionaries[c][word] = 1
                if word in self.global_vocab:
                    self.global_vocab.add(word)

        self.prob_w_given_class['pos'] = {}
        self.prob_w_given_class['neg'] = {}

        # Regner ut sannsynligheten for hvert ord gitt klasse
        for word in class_dictionaries['pos']:
            self.prob_w_given_class['pos'][word] = math.log((class_dictionaries['pos'].get(word, 0.0) + 1) /
                                                            (sum(class_dictionaries['pos'].values()) + len(self.global_vocab)))
        for word in class_dictionaries['neg']:
            self.prob_w_given_class['neg'][word] = math.log((class_dictionaries['neg'].get(word, 0.0) + 1) /
                                                            (sum(class_dictionaries['neg'].values()) + len(self.global_vocab)))

    #Lager en prediction på hver review i X på om den er positiv eller negativ.
    def predict(self, X):
        result = []

        for x in X:
            vocab = self.word_count(self.text_to_words(x))

            pos_score = 0
            neg_score = 0

            for word in vocab:
                pos_score += self.prob_w_given_class['pos'].get(word, 0.0)
                neg_score += self.prob_w_given_class['neg'].get(word, 0.0)

            pos_score += self.class_priors['pos']
            neg_score += self.class_priors['neg']

            if(pos_score > neg_score):
                result.append(1)
            else:
                result.append(0)
        return result

    # Tar en reviewtekst, fjerner punctuation og returnerer en array med alle ordene den inneholder.
    def text_to_words(self, review_text):
        translator = str.maketrans("", "", string.punctuation)
        review_text = review_text.lower()
        review_text = review_text.translate(translator)

        return review_text.split()

    # Tar en string med ord, returnerer et vocabulary med hvert unike ord og antall ganger det finnes i strengen.
    def word_count(self, words):
        dictionary = {}
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
        return dictionary