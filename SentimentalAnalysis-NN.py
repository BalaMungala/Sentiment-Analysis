import pandas as pd
import re
import math
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier



#importing dataset
dataset=pd.read_csv('I://3//Projects//Business//Sentiment Analysis-python//Restaurant_reviews.tsv',delimiter='\t',quoting=3)

#Data preprocessing phase
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#training neural networks
nn_classifier=Sequential()
nn_classifier.add(Dense(input_dim=1500,units=750,activation='relu',kernel_initializer='uniform'))
nn_classifier.add(Dropout(rate=0.2))
nn_classifier.add(Dense(units=750,activation='relu',kernel_initializer='uniform'))
nn_classifier.add(Dropout(rate=0.2))
nn_classifier.add(Dense(units=750,activation='relu',kernel_initializer='uniform'))
nn_classifier.add(Dropout(rate=0.2))
nn_classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
nn_classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
path='I://3//Projects//Business//Sentiment Analysis-python//sent_weights//weights.{epoch:02d}-{loss:.2f}.hdf5'    
mcp=ModelCheckpoint(path,monitor='loss',save_best_only=True,verbose=0)
nn_classifier.fit(x_train,y_train,epochs=100,batch_size=30,callbacks=[mcp])
sent_pred=nn_classifier.predict(x_test)   
test_set=(sent_pred>0.5)
nn_cm=confusion_matrix(y_test,test_set)
nn_accuracy=(nn_cm[0,0]+nn_cm[1,1])/(nn_cm[0,0]+nn_cm[1,1]+nn_cm[0,1]+nn_cm[1,0])*100
nn_precision=(nn_cm[0,0])/(nn_cm[0,0]+nn_cm[0,1])
nn_recall=(nn_cm[0,0])/(nn_cm[0,0]+nn_cm[1,0])
nn_f1_score=(2*nn_precision*nn_recall)/(nn_precision+nn_recall)
nn_fpr,nn_tpr,nn_threshold=roc_curve(test_set,y_test)
nn_roc_auc = auc(nn_fpr,nn_tpr)

print("Accuracy of NN is {}%".format(math.floor(nn_accuracy)))
print("Precision of NN is {}%".format(math.floor(nn_precision*100)))
print("Recall of NN is {}%".format(math.floor(nn_recall*100)))
print("F1_score of NN is {}%".format(math.floor(nn_f1_score*100)))
print("ROC_curve of NN is {}%".format(math.floor(nn_roc_auc*100)))

#nb_roc_curve graph
plt.title('NN Receiver Operating Characteristic')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(nn_fpr,nn_tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.savefig('I://3//Projects//Business//Sentiment Analysis-python//NN.png')



#cross validation data
def build_classifier():
    nn_classifier=Sequential()
    nn_classifier.add(Dense(input_dim=1500,units=750,activation='relu',kernel_initializer='uniform'))
    nn_classifier.add(Dropout(rate=0.2))
    nn_classifier.add(Dense(units=750,activation='relu',kernel_initializer='uniform'))
    nn_classifier.add(Dropout(rate=0.2))
    nn_classifier.add(Dense(units=750,activation='relu',kernel_initializer='uniform'))
    nn_classifier.add(Dropout(rate=0.2))
    nn_classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
    nn_classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return nn_classifier


classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

#grid search cv
def build_classifier(optimizer):
    nn_classifier=Sequential()
    nn_classifier.add(Dense(input_dim=1500,units=750,activation='relu',kernel_initializer='uniform'))
    nn_classifier.add(Dropout(rate=0.2))
    nn_classifier.add(Dense(units=750,activation='relu',kernel_initializer='uniform'))
    nn_classifier.add(Dropout(rate=0.2))
    nn_classifier.add(Dense(units=750,activation='relu',kernel_initializer='uniform'))
    nn_classifier.add(Dropout(rate=0.2))
    nn_classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
    nn_classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return nn_classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters={'optimizer':['adam','rmsprop'],'batch_size':[20,30,40],'epochs':[80,100,120]}
gscv=GridSearchCV(estimator=classifier,param_grid=parameters,cv=10,scoring='accuracy')
gscv_model=gscv.fit(x_train,y_train)
gscv.best_params_
gscv.best_score_



#Deploying model
nn_classifier=Sequential()
nn_classifier.add(Dense(input_dim=1500,units=750,activation='relu',kernel_initializer='uniform'))
nn_classifier.add(Dropout(rate=0.2))
nn_classifier.add(Dense(units=750,activation='relu',kernel_initializer='uniform'))
nn_classifier.add(Dropout(rate=0.2))
nn_classifier.add(Dense(units=750,activation='relu',kernel_initializer='uniform'))
nn_classifier.add(Dropout(rate=0.2))
nn_classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
nn_classifier.load_weights("I://3//Projects//Business//Sentiment Analysis-python//weights.48-0.01.hdf5")

#data preprocessing of new predictions
def format_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

new_review ='fodd was awesome'
new_review = format_review(new_review)
test_corpus = []
test_corpus.append(new_review)
X_new_test = cv.transform(test_corpus).toarray()

#predicting new review
predicted_new=nn_classifier.predict(X_new_test)
if(predicted_new>0.5):
    print("FOOD IS GOOD")
else:
    print("FOOD IS BAD")    
    