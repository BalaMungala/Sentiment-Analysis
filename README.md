# SentimentAnalysis-Restaurant_Reviews
Performing Sentiment Analysis using Machine Learning( 6 Algorithms) and Neural Networks  in Python

MACHINE LEARNING:

Performing Sentiment Analysis using SIX Maching Learning Algorithms.They are:

      1) Logistic Regression
      2) K-Nearest Neighbors
      3) Naive-Bayes
      4) Support Vector Machine
      5) DecisionTree
      6) Random Forest
      
By training with above Algorithms , we have to choose best Algorithm based on:

      1) Accuracy
      2) Precision
      3) Recall
      4) f1_score
      5) auc(Area under curve)
      
we do not finalize the model based on one test set, so perform "CROSS VALIDATION" then apply above metrics to evaluate model performance,if model performance is not good then we have to go for "GRID SEARCH CV" which is used for "TUNING PARAMETERS" , by training with different parameters ,we get best parameters based on above metrics ,then keep best suited parameters in choosen algorithm ,then deploy the algorithm for production.


ARTIFICIAL NEURAL NETWORKS:
       Neural Networks Training steps:
       
                  1) Import libraries
                  2) Data Preprocessing
                  3) Create dense input layers with 1500 units
                  4) Create dense hidden layers with 750 units
                  5) Apply backpropagation(optimizer) algorithm ("adam")
                  6) Set epochs count
                  7) Call the callbacks for saving weights
                  8) Fit the data to train the networks
                  9) Calclate accuracy,precision,recall,f1_score,auc using confusion matrix with test data
                  10) Evaluate the model
                  11) Improve the model by using dropout regularization
                  12) Tune the parameters
                  13) Based on above metrics choose the best parameters 
                  14)Train with choosen parameters and deploy the model for production
                  
