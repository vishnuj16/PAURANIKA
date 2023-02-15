from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import nltk
nltk.download('omw-1.4')
import string
import random

#IMPORTING AND READING THE CORPUS FOR THE BOT
f = open('corpus.txt', 'r', errors = 'ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()
nltk.download('punkt') #punkt tokenizer
nltk.download('wordnet') #wordnet dictionary being used
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

# print(word_tokens[:3])
# print(sent_tokens[:4])

#TEXT PREPROCESSING
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict( (ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#DEFINING THE GREETINGS, STARTING THE CONVERSATIONS WITH CHATBOT
GREET_IN  = ("hello", "hi", "greetings", "namaste", "wassup", "hey", "heyo", "sup")
GREET_OUT = ("hello", "hi", "Oh hello", "greetings", "namaste", "vanakkam", "*turning towards you*")

def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_IN:
            return random.choice(GREET_OUT)

#RESPONSE GENERATION
def response(user_response):
    robo1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten() 
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo1_response+= "I am Sorry! I don't Understand you"
        return robo1_response
    else:
        robo1_response+=sent_tokens[idx]
        return robo1_response

#CONVERSATION START/END PROTOCOLS
flag = True
print("[To exit, type bye]\n\nPAURANIKA : Namaste! I am Pauranika. how can I help you today?")
while(flag):
    user_response = input("YOU       : ")
    user_response = user_response.lower()
    if (user_response!= "bye"):
        if (user_response == 'thanks' or user_response == 'thank you' or user_response == 'thx'):
            flag = False
            print('PAURANIKA : You are Welcome')
        else:
            if(greet(user_response)!=None):
                print("PAURANIKA : " + greet(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens+=nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('PAURANIKA : ', end = "")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else : 
        flag = False
        print('PAURANIKA : Goodbye! Have a nice day')




