from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

import sys
reload(sys)
sys.setdefaultencoding('Cp1252')

doc_set=[]

fh=open('wiki_data.txt','r')
count=1
for line in fh:
	temp=""
	print count
	temp=temp+line.split(",,,,")[0].strip()[2:-2].strip()+" "
	temp=temp+line.split(",,,,")[1].strip()[2:-2].strip()
	
	count=count+1
	#print "______________"
	#print line
	#print temp
	#print "***************"
	doc_set.append(temp)

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


# list for tokenized documents in loop
texts = []

# loop through document list
count=1
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    #print "**********"
    #print count
    #print raw

    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1496, id2word = dictionary, passes=10        )

topicList=ldamodel.print_topics(num_topics=1496,num_words=20)
f1=open('output_lda.txt','w')
for i in topicList:
    f1.write(str(i))
f1.close()
#test on a new data
'''
print "\n\nTesting\n\n"
testText="Computing the Camera Motion Direction from Many Images.We analyze the problem of estimating a camera's motion direction from a calibrated multi-image sequence. We assume that the camera moves roughly along a line and that its velocity and orientation are unknown and can vary over time. For infinitesimal camera motion (multiple flows rather than multiple images), we give a closed-form expression for the result of minimizing the true least-squares error over all variables but the camera's motion direction. Our result includes the rigidity constraint that the scene stays fixed over time. For finite motion, we present a noniterative algorithm that approximates the exact multi-image coplanarity error to better than a percent. Also, we define a new error contribution that incorporates the rigidity constraint and is analogous to the rigidity component of the error for infinitesimal motion. By adding this to the coplanarity error, we obtain a noniterative algorithm that approximates the complete finite motion least-squares error--including rigidity--as a function just of the translation direction."

print testText
print "\n\n"

raw=testText.lower()
testTokens=tokenizer.tokenize(raw)
stopped_test_tokens=[i for i in testTokens if not i in en_stop]
stemmped_test_tokens=[p_stemmer.stem(i) for i in stopped_test_tokens]

testText_bow=dictionary.doc2bow(stemmped_test_tokens)
testText_lda=ldamodel[testText_bow]

#print the test data most probable topics and percentage matched
print(testText_lda)
'''
wiki_dict = {}
wiki_file = open("wiki_data.txt",'r')
for line in wiki_file.readlines():
    line = line.split(",,,,")
    line[0] = line[0].replace("'","")
    line[0] = line[0].replace('"','')
    line[1] = line[1].replace("'","")
    line[1] = line[1].replace('"','')
    testText = line[0]+" "+ line[1]
    raw=testText.lower()
    testTokens=tokenizer.tokenize(raw)
    stopped_test_tokens=[i for i in testTokens if not i in en_stop]
    stemmped_test_tokens=[p_stemmer.stem(i) for i in stopped_test_tokens]
    testText_bow=dictionary.doc2bow(stemmped_test_tokens)
    testText_lda=ldamodel[testText_bow]
    #print(testText_lda)
    cluster_no = -1
    maxi = 0
    for i in testText_lda:
        if i[1] > maxi:
            maxi = i[1]
            cluster_no = i[0]
    if cluster_no in wiki_dict and wiki_dict[cluster_no][0] < maxi:
        wiki_dict[cluster_no] = (maxi, line[0])
    elif cluster_no not in wiki_dict:
        wiki_dict[cluster_no] = (maxi,line[0])  

#print wiki_dict
files=["test1.txt","test2.txt","test3.txt","test4.txt","test5.txt","test6.txt","test7.txt","test8.txt","test9.txt","test10.txt"]
for fileName in files:
    with open(fileName, 'r') as content_file:
        testDocument = content_file.read()
    #testDocument = raw_input("Copy your test document here:")
    raw=testDocument.lower()
    testTokens=tokenizer.tokenize(raw)
    stopped_test_tokens=[i for i in testTokens if not i in en_stop]
    stemmped_test_tokens=[p_stemmer.stem(i) for i in stopped_test_tokens]
    testText_bow=dictionary.doc2bow(stemmped_test_tokens)
    testText_lda=ldamodel[testText_bow]
    maxi = 0
    cluster_no = -1
    for i in testText_lda:
            if i[1] > maxi:
                maxi = i[1]
                cluster_no = i[0]
    print wiki_dict[cluster_no][1]
