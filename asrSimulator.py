import os
import re
import random
import pickle
import difflib
import numpy as np
from nltk.corpus import words
from multiprocessing import Pool


class AsrSimulator():
    wordDictionary = None
    lengthWordDictionary = None

    def __init__(self, prob_missingWord=0.0, prob_confusedWord=0.0, prob_confusedWord_uncertainty=0.0, prob_extraWord=0.00, prob_randomPause=0.0):
        #This constructor initializes all the parameters and creates an instance
        self.timestamp = 0
        self.prob_missingWord = prob_missingWord
        self.prob_confusedWord = prob_confusedWord
        self.prob_confusedWord_uncertainty = prob_confusedWord_uncertainty
        self.prob_extraWord = prob_extraWord
        self.prob_randomPause = prob_randomPause

        self.changeScaleOverTime = False  # This simulates that there are multiple people talking, each one with its own rithm
        #These parameters are tuples of mean and std deviation in ms
        self.betweenWordsPauseParams = 78, 1.3
        self.commaPauseParams = 426, 1.6
        self.dotPauseParams = 1585, 1.3
        self.specialActionParams = 100, 50
        self.randomPauseParams = 1200, 1500

        self.dictionary = words.words()
        self.words_length = len(self.dictionary)
        self.precalculateLengthWordDictionary()
        self.precalculateWordDictionary()

        self.letterDuration = np.asarray(pickle.load(open("wordsLetterDurations.pkl", "rb")))

    def is_missWord(self):
        # Evaluates if a word should be missed
        randomVal = random.random()
        return randomVal < self.prob_missingWord, randomVal

    def is_extraWord(self):
        # Evaluates if an extra word should be added
        return random.random() < self.prob_extraWord

    def is_confuseWord(self):
        # Evaluates if a word should be confused with another one
        return random.random() < self.prob_confusedWord

    def is_randomPause(self):
        # Evaluates if a word should be confused with another one
        return random.random() < self.prob_randomPause

    def generateWordsWithProb(self, wordList):
        # Generates Elements set of words with probabilities
        element = "{"
        probabilities = np.random.rand(len(wordList))
        probabilities /= np.sum(probabilities)
        for it, word in enumerate(wordList):
            element += '"' + word + '":' + str(probabilities[it])
            if it + 1 < len(wordList):
                element += ","
        element += "}"
        return element

    def generatePhonems(self, word):
        # Should generate phonems, but outputs the text itself
        element = "{"
        element += '"' + word + '"'
        element += "}"
        return element

    def generateWordDuration(self, word):
        letterDurationIndexes = np.random.choice(len(self.letterDuration), len(word))
        return round(np.sum(self.letterDuration[np.array(letterDurationIndexes)]),2)

    def generateTimestamps(self, duration):
        # Generate the timestamp and duration of a word
        return (str(self.timestamp), str(duration)[:4])

    def pauseBetweenWords(self):
        mu, sigma = self.betweenWordsPauseParams
        return self.generateLogRandomValue(mu, sigma)

    def generateAsrElement(self, wordList, phonemList, duration):
        # Function to create each Asr element that corresponds to a word or token: {words and probabilities}, {phonems}, {duration, timestamp}
        element = self.generateWordsWithProb(wordList)
        element += "," + self.generatePhonems(phonemList)
        timestamps = self.generateTimestamps(duration)
        element += "," + "{" + timestamps[0] + "," + timestamps[1] + "}"
        self.timestamp += duration
        self.timestamp += self.pauseBetweenWords()
        self.timestamp = round(self.timestamp, 2)
        return element

    def generatePSRTElement(self, wordList, phonemList, duration):
        # Function to create each PSRT element that corresponds to a word or token:"word timestamp duration prob"
        element = wordList[0]
        timestamps = self.generateTimestamps(duration)
        element += " " + timestamps[0] + " " + timestamps[1] + " 1.0"
        self.timestamp += duration
        return element

    def generateElement(self, wordList, phonemList, duration):
        return self.generateAsrElement(wordList, phonemList, duration)

    def generateRandomValue(self, mu, sigma):
        # generate random value from normal distribution
        s = np.random.normal(mu, sigma, 1)
        return round(s[0], 2)

    def generateLogRandomValue(self, mu, sigma):
        # generate random value from log normal distribution
        s = np.random.normal(np.log10(mu), np.log10(sigma),1)
        return round(np.power(10, s[0])/1000, 2)

    def missWordElement(self, word):
        # Generate a token for transcription error, due to noise or lack of pronuntiation
        resetTimestamp = False
        if self.timestamp == 0:
            resetTimestamp = True
        self.generateElement(["<Unk>"], word, self.generateWordDuration(word))
        if resetTimestamp:
            self.timestamp = 0

    def specialActionElement(self, specialWord, params):
        # Elements like <Laughter>, <Cough>, <SIL> generate the correspondent Element
        mu, sigma = params[0], params[1]  # mean and standard deviation
        return self.generateElement([specialWord], "", self.generateRandomValue(mu, sigma))

    def commaPauseElement(self, params):
        # Comma generates a pause
        mu, sigma = params  # mean and standard deviation
        return self.generateElement(["<,>"], "", self.generateLogRandomValue(mu, sigma))

    def dotPauseElement(self, params):
        # Dot generates a pause
        mu, sigma = params  # mean and standard deviation
        return self.generateElement(["<.>"], "", self.generateLogRandomValue(mu, sigma))


    def extraWord(self, word):
        # Word generated from splitting one word into two
        firstWordLength = np.random.randint(2, len(word))
        secondWordLength = len(word) - firstWordLength
        firstWord = []
        try:
            while (len(firstWord) == 0 or len(secondWord) == 0) and firstWordLength in self.lengthWordDictionary.keys():
                firstWord = difflib.get_close_matches(word[:firstWordLength], self.lengthWordDictionary[firstWordLength])
                secondWord = difflib.get_close_matches(word[firstWordLength:], self.lengthWordDictionary[secondWordLength])
                if len(firstWord) == 0 or len(secondWord) == 0:
                    firstWordLength += 1
                    secondWordLength = len(word) - firstWordLength
                    if secondWordLength == 0:
                        break
                else:
                    return firstWord[0], secondWord[0]
        except Exception as e:
            print(word)
            print(str(e))
        return word, None

    def randomPause(self, params):
        # Word generated from noise, this is a transcription error
        mu, sigma = params  # mean and standard deviation
        return self.generateElement(["RandomPause"], "", self.generateRandomValue(mu, sigma))

    def wordElement(self, word):
        # Generate a word element
        return self.generateElement([word], word, self.generateWordDuration(word))

    def get_close_matches(self, word):
        #findes close match for given word
        similarWords = difflib.get_close_matches(word, words.words())
        return word, similarWords

    def precalculateWordDictionary(self):
        # precalculate Dictionary, it may take a long time to calculate
        if self.wordDictionary is None:
            it = 0
            dictionaryFileName = "dictionarySimilarWords.p"
            if os.path.isfile(dictionaryFileName):
                self.wordDictionary = pickle.load(open(dictionaryFileName, "rb"))
            else:
                self.wordDictionary = {}
                print("Creating dictionary")
                with Pool() as p:
                    res = p.map(self.get_close_matches, words.words())
                    for element in res:
                        self.wordDictionary[element[0]] = element[1]
                print("Dictionary created")
                pickle.dump(self.wordDictionary, open(dictionaryFileName, "wb"))

    def precalculateLengthWordDictionary(self):
        # precalculate Length Dictionary, it may take a long time to calculate
        if self.lengthWordDictionary is None:
            it = 0
            dictionaryFileName = "dictionaryLengthWords.p"
            if os.path.isfile(dictionaryFileName):
                self.lengthWordDictionary = pickle.load(open(dictionaryFileName, "rb"))
            else:
                self.lengthWordDictionary = {}
                self.lengthWordDictionary[0] = []
                print("Creating lengthWordDictionary")
                for word in words.words():
                    wordLength = len(word)
                    if wordLength not in self.lengthWordDictionary.keys():
                        self.lengthWordDictionary[wordLength] = []
                    if word not in self.lengthWordDictionary[wordLength]:
                        self.lengthWordDictionary[wordLength].append(word)
                print("Dictionary created lengthWordDictionary")
                pickle.dump(self.lengthWordDictionary, open(dictionaryFileName, "wb"))

    def confuseWordElement(self, word):
        # Dot generates a confusion word that is more probable than the correct one
        if word in self.wordDictionary.keys():
            return self.generateElement(self.wordDictionary[word], word, self.generateWordDuration(word))
        else:
            return self.generateElement(["UNK"], word, self.generateWordDuration(word))


    def convertSentenceToAsrFormat(self, sentence):
        # Converts a (label, text) array into ASR format
        asrStream = []
        iteration = 0
        sentence = sentence.replace(":", "")
        sentence = re.sub(' +', ' ', sentence).lstrip()
        sentence = sentence.split(" ")
        missingWordProbabilities = []
        for word in sentence:
            iteration += 1
            hasComma = "," in word
            hasDot = "." in word
            isSpecialAction = "<" in word
            word = word.replace(',', '')
            word = word.replace('.', '')
            word = re.sub(r'[^\w]', ' ', word).rstrip()
            if len(word) > 0:
                is_missWord, missProbability = self.is_missWord()
                if is_missWord:
                    missingWordProbabilities.append(missProbability)
                    self.missWordElement(word)
                elif self.is_extraWord() and len(word) > 2:
                    firstWord, secondWord = self.extraWord(word)
                    if firstWord is not None and secondWord is not None:
                        asrStream.append(self.wordElement(firstWord))
                        asrStream.append(self.wordElement(secondWord))
                    else:
                        asrStream.append(self.wordElement(word))
                elif self.is_confuseWord():
                    asrStream.append(self.confuseWordElement(word))
                else:
                    if isSpecialAction:
                        asrStream.append(self.specialActionElement(word, self.specialActionParams))
                    elif len(word) > 0:
                        asrStream.append(self.wordElement(word))

                # This two elements can insert an element or not
                if hasComma:
                    self.commaPauseElement(self.commaPauseParams)
                if hasDot:
                    self.dotPauseElement(self.dotPauseParams)
                    if self.changeScaleOverTime:
                        self.scale = 0.5 + random.random()

                if self.is_randomPause():
                    self.randomPause(self.randomPauseParams)
            else:
                missingWordProbabilities.append(-1)

        if len(asrStream) == 0 and len(missingWordProbabilities) > 0:
            bestWord = sentence[np.argmax(missingWordProbabilities)]
            bestWord = bestWord.replace(',', '')
            bestWord = bestWord.replace('.', '')
            bestWord = re.sub(r'[^\w]', ' ', bestWord).rstrip()
            if len(bestWord) > 0:
                asrStream.append(self.wordElement(bestWord))
            else:
                print("----##-----", sentence)
        return asrStream  # , endSentenceLabel




if __name__ == "__main__":
    print("##########################################################")
    print("#                       EXAMPLE")
    print("##########################################################")
    sentence = "Maybe what I think Tastee Wheat tasted like actually tasted like oatmeal or tuna fish"

    asrSimulator_noise = AsrSimulator()
    asrText = asrSimulator_noise.convertSentenceToAsrFormat(sentence)

    print(sentence)
    print(asrText)
