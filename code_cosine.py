import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]
files=["test1.txt","test2.txt","test3.txt","test4.txt","test5.txt","test6.txt","test7.txt","test8.txt","test9.txt","test10.txt"]
for file_open in files:
	test_file = open(file_open,"r")
	testDocument = test_file.read()
	wiki_file = open("wiki_data.txt",'r')
	maximum = -10
	doc = ""
	for line in wiki_file.readlines():
		line = line.split(",,,,")
		line[0] = line[0].replace("'","")
		line[0] = line[0].replace('"','')
		line[1] = line[1].replace("'","")
		line[1] = line[1].replace('"','')
		testText = line[0]+" "+ line[1]

		sim = cosine_sim(testText, testDocument)
	#print sim
		if sim > maximum:
			maximum = sim
			doc = line[0]
	#print maximum 
	print doc
#print cosine_sim('a little bird', 'a big dog barks')
