
# Trenger sannsynlighetene for:
#   - Klasse(pos eller neg)
#   - Ord
#   - Ord gitt klasse


# Parser alle dokumentene i en mappe og returnerer en liste med alle ordene de inneholder.
def wordParser():
    import os
    import string

    a = list()
    translator = str.maketrans('', '', string.punctuation)
    path = 'C:\\Users\\tordk\\Documents\\INFO284\\1rstGroupAssingment\\Project\\DATA\\aclImdb\\train\\pos'
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
def dictionary():
    counts = dict()
    a = wordParser()
    for word in a:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

# Returnerer antall ord totalt.
def wordTotal():
    bagofwords = wordParser()
    return bagofwords.__len__()

# Returnerer sannsynligheten av noe... nk
def wordProbability(word):
    numofwords = wordTotal()
    dict = dictionary()

    probability = (dictionary[word] + 1) / (numofwords + dict.__len__())
    return probability
