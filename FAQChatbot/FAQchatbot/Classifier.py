import ScrapeForDataset as scraper
import wordVectorRepresentation as wordvec
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import spatial
import xlrd
import math

stopwords = set(stopwords.words('english'))


def getIDF(TrainingSet):
    vectorizer = TfidfVectorizer(min_df=1,stop_words=stopwords)
    lemmatizer = WordNetLemmatizer()

    vectorizer.fit_transform(TrainingSet.Question)
    tfvalues=[lemmatizer.lemmatize(x) for x in list(vectorizer.vocabulary_.keys())]# vectorizer.vocabulary_
    idfvalues = vectorizer.idf_
    IDFset=dict(zip(tfvalues,idfvalues))

    return IDFset


TrainingSet=scraper.createTrainingData()

IDFset=getIDF(TrainingSet)


Question_vectors=wordvec.questiontoVector(TrainingSet,IDFset)



# score = spatial.distance.cosine(query_vector, vec)
#utterence="Who should recommendations come from"

def getAnswer(utterence):

    y=wordvec.vectorize_query(utterence,IDFset)
    scores=[]
    for i in range(Question_vectors.__len__()):
        x = np.array(Question_vectors[i])
        score = spatial.distance.cosine(x, y)
        if math.isnan(score):
            scores.append(1)
        else:
            scores.append(score)


    index=scores.index(np.min(scores))
    print("Index",index)
    answer=TrainingSet.Answer[index]

    return answer


def evaluateAnswers():
    questions=pd.read_excel("Test Questions.xlsx")
    Testquestion = []
    Testanswer = []
    for i in range(questions.__len__()):
        utterence = questions.values[i].item()
        answer = getAnswer(utterence)
        Testanswer.append(answer)
        Testquestion.append(utterence)

    TestSet = pd.DataFrame(list(zip(Testquestion, Testanswer)), columns=['Question', 'Answer'])
    TestSet.to_excel("TestSet3.xlsx")

    return TestSet

evaluateAnswers()