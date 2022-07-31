import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
true_df = pd.read_csv('./True.csv')
fake_df = pd.read_csv('./Fake.csv')
true_df['label'] = 0
fake_df['label'] = 1
true_df = true_df[['text','label']]
fake_df = fake_df[['text','label']]
dataset = pd.concat([true_df , fake_df])
print(dataset['label'].value_counts())
dataset = dataset.sample(frac = 1)
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
def clean_data(text):
    text = text.lower() 
    text = re.sub('[^a-zA-Z]' , ' ' , text)
    token = text.split() 
    news = [lemmatizer.lemmatize(word) for word in token if not word in stopwords]  
    clean_news = ' '.join(news) 
    
    return clean_news 

dataset['text'] = dataset['text'].apply(lambda x : clean_data(x))
vectorizer = TfidfVectorizer(max_features = 50000 , lowercase=False , ngram_range=(1,2))
X = dataset.iloc[:20000,0]
y = dataset.iloc[:20000,1]
train_X , test_X , train_y , test_y = train_test_split(X , y , test_size = 0.2 ,random_state = 0)
vec_train = vectorizer.fit_transform(train_X)
vec_train = vec_train.toarray()
vec_test = vectorizer.transform(test_X).toarray()
train_data = pd.DataFrame(vec_train , columns=vectorizer.get_feature_names_out())
test_data = pd.DataFrame(vec_test , columns= vectorizer.get_feature_names_out())
level0 = list()
level0.append(('rf',RandomForestClassifier()))
level0.append(('dt',DecisionTreeClassifier()))
level1 = MultinomialNB()
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
model.fit(train_data, train_y)
predictions = model.predict(test_data)
print(classification_report(test_y , predictions))


