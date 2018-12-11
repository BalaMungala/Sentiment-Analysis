#importing libraries
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

#Training the model phase - Training with all Algorithms

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB(priors=None)
nb_classifier=nb.fit(x_train,y_train)
nb_y_pred=nb_classifier.predict(x_test)
nb_cm=confusion_matrix(y_test,nb_y_pred)
nb_accuracy=(nb_cm[0,0]+nb_cm[1,1])/(nb_cm[0,0]+nb_cm[1,1]+nb_cm[0,1]+nb_cm[1,0])*100
nb_precision=(nb_cm[0,0])/(nb_cm[0,0]+nb_cm[0,1])
nb_recall=(nb_cm[0,0])/(nb_cm[0,0]+nb_cm[1,0])
nb_f1_score=(2*nb_precision*nb_recall)/(nb_precision+nb_recall)
nb_fpr,nb_tpr,nb_threshold=roc_curve(nb_y_pred,y_test)
nb_roc_auc = auc(nb_fpr,nb_tpr)

print("Accuracy of Naive_Bayes is {}%".format(math.floor(nb_accuracy)))
print("Precision of Naive_Bayes is {}%".format(math.floor(nb_precision*100)))
print("Recall of Naive_Bayes is {}%".format(math.floor(nb_recall*100)))
print("F1_score of Naive_Bayes is {}%".format(math.floor(nb_f1_score*100)))
print("ROC_curve of Naive_Bayes is {}%".format(math.floor(nb_roc_auc*100)))


#evaluating the accuracy,precision,recall,f1score with 10 folds(with 10 test-data)
nb_accuracies=cross_val_score(estimator=nb,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='accuracy')
nb_avg_accuracy=math.floor((nb_accuracies.mean())*100)
nb_std_accuracies=math.floor(nb_accuracies.std()*100)

nb_precisions=cross_val_score(estimator=nb,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='precision')
nb_avg_precision=math.floor((nb_precisions.mean())*100)
nb_std_precision=math.floor(nb_precisions.std()*100)

nb_recalls=cross_val_score(estimator=nb,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='recall')
nb_avg_recall=math.floor((nb_recalls.mean())*100)
nb_std_recall=math.floor(nb_recalls.std()*100)

nb_f1s=cross_val_score(estimator=nb,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='f1')
nb_avg_f1s=math.floor((nb_f1s.mean())*100)
nb_std_f1s=math.floor(nb_f1s.std()*100)

nb_roc_auc=cross_val_score(estimator=nb,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='roc_auc')
nb_avg_roc_auc=math.floor((nb_roc_auc.mean())*100)
nb_std_roc_auc=math.floor(nb_roc_auc.std()*100)


#nb_roc_curve graph
plt.title('NB Receiver Operating Characteristic')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(nb_fpr,nb_tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.savefig('I://3//Projects//Business//Sentiment Analysis-python//nb_roc.png')


#tuning the parameters
"""
no parameters are present in class , so no need of tuning the parameters
"""


#Logistic Regression
from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression(penalty='l2',
                       dual=False,
                       tol=1e-4,
                       C=1.0, 
                       fit_intercept=True,
                       intercept_scaling=1,
                       class_weight=None,
                       random_state=0,
                       solver='liblinear',
                       max_iter=100,
                       multi_class='ovr', 
                       verbose=0,
                       warm_start=False,
                       n_jobs=1)
lgr_classifier=lgr.fit(x_train,y_train)
lgr_y_pred=lgr_classifier.predict(x_test)
lgr_cm=confusion_matrix(y_test,lgr_y_pred)
lgr_accuracy=(lgr_cm[0,0]+lgr_cm[1,1])/(lgr_cm[0,0]+lgr_cm[1,1]+lgr_cm[0,1]+lgr_cm[1,0])*100
lgr_precision=(lgr_cm[0,0])/(lgr_cm[0,0]+lgr_cm[0,1])
lgr_recall=(lgr_cm[0,0])/(lgr_cm[0,0]+lgr_cm[1,0])
lgr_f1_score=(2*lgr_precision*lgr_recall)/(lgr_precision+lgr_recall)
lgr_fpr,lgr_tpr,lgr_threshold=roc_curve(lgr_y_pred,y_test)
lgr_roc_auc = auc(lgr_fpr,lgr_tpr)

print("Accuracy of Logistic-Regression is {}%".format(math.floor(lgr_accuracy)))
print("Precision of Logistic-Regression is {}%".format(math.floor(lgr_precision*100)))
print("Recall of Logistic-Regression is {}%".format(math.floor(lgr_recall*100)))
print("F1_score of Logistic-Regression is {}%".format(math.floor(lgr_f1_score*100)))
print("ROC_curve of Logistic-Regression is {}%".format(math.floor(lgr_roc_auc*100)))

#evaluating the accuracy,precision,recall,f1score with 10 folds(with 10 test-data)
lgr_accuracies=cross_val_score(estimator=lgr,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='accuracy')
lgr_avg_accuracy=math.floor((lgr_accuracies.mean())*100)
lgr_std_accuracies=math.floor(lgr_accuracies.std()*100)

lgr_precisions=cross_val_score(estimator=lgr,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='precision')
lgr_avg_precision=math.floor((lgr_precisions.mean())*100)
lgr_std_precision=math.floor(lgr_precisions.std()*100)

lgr_recalls=cross_val_score(estimator=lgr,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='recall')
lgr_avg_recall=math.floor((lgr_recalls.mean())*100)
lgr_std_recall=math.floor(lgr_recalls.std()*100)

lgr_f1s=cross_val_score(estimator=lgr,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='f1')
lgr_avg_f1s=math.floor((lgr_f1s.mean())*100)
lgr_std_f1s=math.floor(lgr_f1s.std()*100)

lgr_roc_auc=cross_val_score(estimator=lgr,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='roc_auc')
lgr_avg_roc_auc=math.floor((lgr_roc_auc.mean())*100)
lgr_std_roc_auc=math.floor(lgr_roc_auc.std()*100)

#lgr_roc_curve graph
plt.title("LGR Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(lgr_fpr,lgr_tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.savefig('I://3//Projects//Business//Sentiment Analysis-python//lgr_roc.png')


#tuning the parameters for improving the accuracy,precision,recall,f1_score
lgr.get_params()
lgr_parameters={'C': [1.0,0.8,0.6],
 'max_iter': [80,100,120],
 'penalty': ['l2','l1'],
 }
for scor in ['accuracy','precision','recall','f1'] :
    lgr_grid_cv_prc=GridSearchCV(estimator=lgr,param_grid=lgr_parameters,scoring=scor,n_jobs=-1,cv=10)
    lgr_grid_model_prc=lgr_grid_cv_prc.fit(x_train,y_train)
    print(scor+" is",lgr_grid_model_prc.best_score_)
    print(scor+" params are",lgr_grid_cv_prc.best_params_)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="gini",
                           splitter="best", 
                           max_depth=None, 
                           min_samples_split=2, 
                           min_samples_leaf=1,
                           min_weight_fraction_leaf=0.,
                           max_features=None,
                           random_state=0,
                           max_leaf_nodes=None,
                           min_impurity_decrease=0., 
                           min_impurity_split=None, 
                           class_weight=None,
                           presort=False)
dtc_classifier=dtc.fit(x_train,y_train)
dtc_y_pred=dtc_classifier.predict(x_test)
dtc_cm=confusion_matrix(y_test,dtc_y_pred)
dtc_accuracy=(dtc_cm[0,0]+dtc_cm[1,1])/(dtc_cm[0,0]+dtc_cm[1,1]+dtc_cm[0,1]+dtc_cm[1,0])*100
dtc_precision=(dtc_cm[0,0])/(dtc_cm[0,0]+dtc_cm[0,1])
dtc_recall=(dtc_cm[0,0])/(dtc_cm[0,0]+dtc_cm[1,0])
dtc_f1_score=(2*dtc_precision*dtc_recall)/(dtc_precision+dtc_recall)
dtc_fpr,dtc_tpr,dtc_threshold=roc_curve(dtc_y_pred,y_test)
dtc_roc_auc = auc(dtc_fpr,dtc_tpr)

print("Accuracy of Decision-Tree is {}%".format(math.floor(dtc_accuracy)))
print("Precision of Decision-Tree is {}%".format(math.floor(dtc_precision*100)))
print("Recall of Decision-Tree is {}%".format(math.floor(dtc_recall*100)))
print("F1_score of Decision-Tree is {}%".format(math.floor(dtc_f1_score*100)))
print("ROC_curve of Decision-Tree is {}%".format(math.floor(dtc_roc_auc*100)))

#evaluating the accuracy,precision,recall,f1score with 10 folds(with 10 test-data)
dtc_accuracies=cross_val_score(estimator=dtc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='accuracy')
dtc_avg_accuracy=math.floor((dtc_accuracies.mean())*100)
dtc_std_accuracies=math.floor(dtc_accuracies.std()*100)

dtc_precisions=cross_val_score(estimator=dtc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='precision')
dtc_avg_precision=math.floor((dtc_precisions.mean())*100)
dtc_std_precision=math.floor(dtc_precisions.std()*100)

dtc_recalls=cross_val_score(estimator=dtc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='recall')
dtc_avg_recall=math.floor((dtc_recalls.mean())*100)
dtc_std_recall=math.floor(dtc_recalls.std()*100)

dtc_f1s=cross_val_score(estimator=dtc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='f1')
dtc_avg_f1s=math.floor((dtc_f1s.mean())*100)
dtc_std_f1s=math.floor(dtc_f1s.std()*100)

dtc_roc_auc=cross_val_score(estimator=dtc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='roc_auc')
dtc_avg_roc_auc=math.floor((dtc_roc_auc.mean())*100)
dtc_std_roc_auc=math.floor(dtc_roc_auc.std()*100)


#tuning the parameters for improving the accuracy,precision,recall,f1_score
dtc_parameters={'criterion':['gini','entropy']}
for scor in ['accuracy','precision','recall','f1'] :
    dtc_grid_cv_prc=GridSearchCV(estimator=dtc,param_grid=dtc_parameters,scoring=scor,n_jobs=-1,cv=10)
    dtc_grid_model_prc=dtc_grid_cv_prc.fit(x_train,y_train)
    print(scor+" is",dtc_grid_model_prc.best_score_)
    print(scor+" params are",dtc_grid_cv_prc.best_params_)

#dtc_roc_curve graph
plt.title("DTC Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(dtc_fpr,dtc_tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.savefig('I://3//Projects//Business//Sentiment Analysis-python//dtc_roc.png')

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=20, 
                           criterion="gini", 
                           max_depth=None,
                           min_samples_split=2, 
                           min_samples_leaf=1,
                           min_weight_fraction_leaf=0.,
                           max_features="auto",
                           max_leaf_nodes=None, 
                           min_impurity_decrease=0.,
                           min_impurity_split=None,
                           bootstrap=True, 
                           oob_score=False,
                           n_jobs=-1,
                           random_state=0, 
                           verbose=0, 
                           warm_start=False, 
                           class_weight=None)
rfc_classifier=rfc.fit(x_train,y_train)
rfc_y_pred=rfc_classifier.predict(x_test)
rfc_cm=confusion_matrix(y_test,rfc_y_pred)
rfc_accuracy=(rfc_cm[0,0]+rfc_cm[1,1])/(rfc_cm[0,0]+rfc_cm[1,1]+rfc_cm[0,1]+rfc_cm[1,0])*100
rfc_precision=(rfc_cm[0,0])/(rfc_cm[0,0]+rfc_cm[0,1])
rfc_recall=(rfc_cm[0,0])/(rfc_cm[0,0]+rfc_cm[1,0])
rfc_f1_score=(2*rfc_precision*rfc_recall)/(rfc_precision+rfc_recall)
rfc_fpr,rfc_tpr,rfc_threshold=roc_curve(rfc_y_pred,y_test)
rfc_roc_auc = auc(rfc_fpr,rfc_tpr)

print("Accuracy of Random-Forest is {}%".format(math.floor(rfc_accuracy)))
print("Precision of Random-Forest is {}%".format(math.floor(rfc_precision*100)))
print("Recall of Random-Forest is {}%".format(math.floor(rfc_recall*100)))
print("F1_score of Random-Forest is {}%".format(math.floor(rfc_f1_score*100)))
print("ROC_curve of Random-Forest is {}%".format(math.floor(rfc_roc_auc*100)))



#evaluating the accuracy,precision,recall,f1score with 10 folds(with 10 test-data)
rfc_accuracies=cross_val_score(estimator=rfc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='accuracy')
rfc_avg_accuracy=math.floor((rfc_accuracies.mean())*100)
rfc_std_accuracies=math.floor(rfc_accuracies.std()*100)

rfc_precisions=cross_val_score(estimator=rfc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='precision')
rfc_avg_precision=math.floor((rfc_precisions.mean())*100)
rfc_std_precision=math.floor(rfc_precisions.std()*100)

rfc_recalls=cross_val_score(estimator=rfc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='recall')
rfc_avg_recall=math.floor((rfc_recalls.mean())*100)
rfc_std_recall=math.floor(rfc_recalls.std()*100)

rfc_f1s=cross_val_score(estimator=rfc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='f1')
rfc_avg_f1s=math.floor((rfc_f1s.mean())*100)
rfc_std_f1s=math.floor(rfc_f1s.std()*100)

rfc_roc_auc=cross_val_score(estimator=rfc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='roc_auc')
rfc_avg_roc_auc=math.floor((rfc_roc_auc.mean())*100)
rfc_std_roc_auc=math.floor(rfc_roc_auc.std()*100)



#tuning the parameters for improving the accuracy,precision,recall,f1_score
rfc_parameters={'n_estimators':[10,15,20,25],'criterion':['gini','entropy']}
for scor in ['accuracy','precision','recall','f1'] :
    rfc_grid_cv_prc=GridSearchCV(estimator=rfc,param_grid=rfc_parameters,scoring=scor,n_jobs=-1,cv=10)
    rfc_grid_model_prc=rfc_grid_cv_prc.fit(x_train,y_train)
    print(scor+" is",rfc_grid_model_prc.best_score_)
    print(scor+" params are",rfc_grid_cv_prc.best_params_)

#rfc_roc_curve graph
plt.title("RFC Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(rfc_fpr,rfc_tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.savefig('I://3//Projects//Business//Sentiment Analysis-python//rfc_roc.png')

#Support Vector Machine
from sklearn.svm import SVC
svc=SVC(C=1.0, 
        kernel='rbf', 
        degree=3, 
        gamma='auto',
        coef0=0.0,
        shrinking=True, 
        probability=False,
        tol=1e-3,
        cache_size=200, 
        class_weight=None,
        verbose=False, 
        max_iter=-1, 
        decision_function_shape='ovr', 
        random_state=0)
svc_classifier=svc.fit(x_train,y_train)
svc_y_pred=svc_classifier.predict(x_test)
svc_cm=confusion_matrix(y_test,svc_y_pred)
svc_accuracy=(svc_cm[0,0]+svc_cm[1,1])/(svc_cm[0,0]+svc_cm[1,1]+svc_cm[0,1]+svc_cm[1,0])*100
svc_precision=(svc_cm[0,0])/(svc_cm[0,0]+svc_cm[0,1])
svc_recall=(svc_cm[0,0])/(svc_cm[0,0]+svc_cm[1,0])
svc_f1_score=(2*svc_precision*svc_recall)/(svc_precision+svc_recall)
svc_fpr,svc_tpr,svc_threshold=roc_curve(svc_y_pred,y_test)
svc_roc_auc = auc(svc_fpr,svc_tpr)

print("Accuracy of Support-Vector-Machine is {}%".format(math.floor(svc_accuracy)))
print("Precision of Support-Vector-Machine is {}%".format(math.floor(svc_precision*100)))
print("Recall of Support-Vector-Machine is {}%".format(math.floor(svc_recall*100)))
print("F1_score of Support-Vector-Machine is {}%".format(math.floor(svc_f1_score*100)))
print("ROC_curve of Support-Vector-Machine is {}%".format(math.floor(svc_roc_auc*100)))

#evaluating the accuracy,precision,recall,f1score with 10 folds(with 10 test-data)
svc_accuracies=cross_val_score(estimator=svc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='accuracy')
svc_avg_accuracy=math.floor((svc_accuracies.mean())*100)
svc_std_accuracies=math.floor(svc_accuracies.std()*100)

svc_precisions=cross_val_score(estimator=svc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='precision')
svc_avg_precision=math.floor((svc_precisions.mean())*100)
svc_std_precision=math.floor(svc_precisions.std()*100)

svc_recalls=cross_val_score(estimator=svc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='recall')
svc_avg_recall=math.floor((svc_recalls.mean())*100)
svc_std_recall=math.floor(svc_recalls.std()*100)

svc_f1s=cross_val_score(estimator=nb,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='f1')
svc_avg_f1s=math.floor((svc_f1s.mean())*100)
svc_std_f1s=math.floor(svc_f1s.std()*100)

svc_roc_auc=cross_val_score(estimator=svc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='roc_auc')
svc_avg_roc_auc=math.floor((svc_roc_auc.mean())*100)
svc_std_roc_auc=math.floor(svc_roc_auc.std()*100)


#tuning the parameters for improving the accuracy,precision,recall,f1_score
svc_parameters={'C':[0.6,0.8,1.0], 
        'kernel':['rbf','linear','poly','sigmoid'], 
        'degree':[3,2,4]}
for scor in ['accuracy','precision','recall','f1'] :
    svc_grid_cv_prc=GridSearchCV(estimator=svc,param_grid=svc_parameters,scoring=scor,n_jobs=-1,cv=10)
    svc_grid_model_prc=svc_grid_cv_prc.fit(x_train,y_train)
    print(scor+" is",svc_grid_model_prc.best_score_)
    print(scor+" params are",svc_grid_cv_prc.best_params_)

#svc_roc_curve graph
plt.title("SVC Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(svc_fpr,svc_tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.savefig('I://3//Projects//Business//Sentiment Analysis-python//svc_roc.png')


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=10,
                         weights='uniform',
                         algorithm='auto', 
                         leaf_size=30,
                         p=2,
                         metric='minkowski'
                         , metric_params=None,
                         n_jobs=-1)
knc_classifier=nb.fit(x_train,y_train)
knc_y_pred=nb_classifier.predict(x_test)
knc_cm=confusion_matrix(y_test,knc_y_pred)
knc_accuracy=(knc_cm[0,0]+knc_cm[1,1])/(knc_cm[0,0]+knc_cm[1,1]+knc_cm[0,1]+knc_cm[1,0])*100
knc_precision=(knc_cm[0,0])/(knc_cm[0,0]+knc_cm[0,1])
knc_recall=(knc_cm[0,0])/(knc_cm[0,0]+knc_cm[1,0])
knc_f1_score=(2*knc_precision*knc_recall)/(knc_precision+knc_recall)
knc_fpr,knc_tpr,knc_threshold=roc_curve(knc_y_pred,y_test)
knc_roc_auc = auc(knc_fpr,knc_tpr)

print("Accuracy of K-Nearest-Neighbors is {}%".format(math.floor(knc_accuracy)))
print("Precision of K-Nearest-Neighbors is {}%".format(math.floor(knc_precision*100)))
print("Recall of K-Nearest-Neighbors is {}%".format(math.floor(knc_recall*100)))
print("F1_score of K-Nearest-Neighbors is {}%".format(math.floor(knc_f1_score*100)))
print("ROC_curve of K-Nearest-Neighbors is {}%".format(math.floor(knc_roc_auc*100)))

#evaluating the accuracy,precision,recall,f1score with 10 folds(with 10 test-data)
knc_accuracies=cross_val_score(estimator=knc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='accuracy')
knc_avg_accuracy=math.floor((knc_accuracies.mean())*100)
knc_std_accuracies=math.floor(knc_accuracies.std()*100)

knc_precisions=cross_val_score(estimator=knc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='precision')
knc_avg_precision=math.floor((knc_precisions.mean())*100)
knc_std_precision=math.floor(knc_precisions.std()*100)

knc_recalls=cross_val_score(estimator=knc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='recall')
knc_avg_recall=math.floor((knc_recalls.mean())*100)
knc_std_recall=math.floor(knc_recalls.std()*100)

knc_f1s=cross_val_score(estimator=knc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='f1')
knc_avg_f1s=math.floor((knc_f1s.mean())*100)
knc_std_f1s=math.floor(knc_f1s.std()*100)

knc_roc_auc=cross_val_score(estimator=knc,X=x_train,y=y_train,cv=10,n_jobs=-1,scoring='roc_auc')
knc_avg_roc_auc=math.floor((knc_roc_auc.mean())*100)
knc_std_roc_auc=math.floor(knc_roc_auc.std()*100)

#tuning the parameters for improving the accuracy,precision,recall,f1_score
knc_parameters={'n_neighbors':[5,10,15]}
for scor in ['accuracy','precision','recall','f1'] :
    knc_grid_cv_prc=GridSearchCV(estimator=knc,param_grid=knc_parameters,scoring=scor,n_jobs=-1,cv=10)
    knc_grid_model_prc=knc_grid_cv_prc.fit(x_train,y_train)
    print(scor+" is",knc_grid_model_prc.best_score_)
    print(scor+" params are",knc_grid_cv_prc.best_params_)

#knc_roc_curve graph
plt.title("KNC Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(knc_fpr,knc_tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.savefig('I://3//Projects//Business//Sentiment Analysis-python//knc_roc.png')

#full roc curves
plt.title("FULL_ROC Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(lgr_fpr,lgr_tpr,'b')
plt.plot(knc_fpr,knc_tpr,'g')
plt.plot(nb_fpr,nb_tpr,'y')
plt.plot(dtc_fpr,dtc_tpr,'m')
plt.plot(rfc_fpr,rfc_tpr,'r')
plt.plot([0,1],[0,1],'r--')
plt.savefig('I://3//Projects//Business//Sentiment Analysis-python//full_roc.png')




#creating datafarme of cross validation metrics for all above algorithms
features_df={'NB':[nb_avg_accuracy,nb_avg_precision,nb_avg_recall,nb_avg_f1s,nb_avg_roc_auc],
             'LGR':[lgr_avg_accuracy,lgr_avg_precision,lgr_avg_recall,lgr_avg_f1s,lgr_avg_roc_auc],
             'DTC':[dtc_avg_accuracy,dtc_avg_precision,dtc_avg_recall,dtc_avg_f1s,dtc_avg_roc_auc],
             'RFC':[rfc_avg_accuracy,rfc_avg_precision,rfc_avg_recall,rfc_avg_f1s,rfc_avg_roc_auc],
             'SVC':[svc_avg_accuracy,svc_avg_precision,svc_avg_recall,svc_avg_f1s,svc_avg_roc_auc],
             'KNN':[knc_avg_accuracy,knc_avg_precision,knc_avg_recall,knc_avg_f1s,knc_avg_roc_auc]}
metrics=pd.DataFrame(features_df,index=['ACCURACY','PRECISION','RECALL','F1_SCORE','ROC_AUC'])

#creating dataframe for grid search metrics

   
#Analysing the metrics
min_accuracy=min(metrics.iloc[0].values)
max_accuracy=max(metrics.iloc[0].values)
print('max_accuracy',max_accuracy)

min_precision=min(metrics.iloc[1].values)
max_precision=max(metrics.iloc[1].values)
print('max_precision',max_precision)
min_recall=min(metrics.iloc[2].values)
max_recall=max(metrics.iloc[2].values)
print('max_recall:',max_recall)

min_f1_score=min(metrics.iloc[3].values)
max_f1_score=max(metrics.iloc[3].values)
print('max_f1_score:',max_f1_score)

min_roc_auc=min(metrics.iloc[4].values)
max_roc_auc=max(metrics.iloc[4].values)
print('max_roc_auc:',max_roc_auc)

print("RESULT:By Analysing above metrics , LOGISTIC REGRESSION is best Algoruthm for this data,so deploying model using Logistic regression gives excellent results")

#Deploying and predicting new review
#model creation
lgr_deploy=LogisticRegression(penalty='l2',
                       dual=False,
                       tol=1e-4,
                       C=1.0, 
                       fit_intercept=True,
                       intercept_scaling=1,
                       class_weight=None,
                       random_state=0,
                       solver='liblinear',
                       max_iter=100,
                       multi_class='ovr', 
                       verbose=0,
                       warm_start=False,
                       n_jobs=1)
lgr_classifier_deploy=lgr_deploy.fit(x_train,y_train)
#preprocessing for the new feature
def format_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

new_review = 'The food was good'
new_review = format_review(new_review)
test_corpus = []
test_corpus.append(new_review)
X_new_test = cv.transform(test_corpus).toarray()

#predicting new review
prediction = lgr_classifier.predict(X_new_test)
if (prediction == 0):
    print('FOOD IS BAD')
elif (prediction == 1):
    print('FODD IS GOOD')    
else:
    print('UNABLE TO PREDICT')    
