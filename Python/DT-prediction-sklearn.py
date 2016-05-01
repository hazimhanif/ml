from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

training = pd.read_csv(open("C:/Users/Hazim/Downloads/training_set.csv"))
testing = pd.read_csv(open("C:/Users/Hazim/Downloads/test_set.csv"))

training_data = training.ix[:,0:2]
training_target = training.ix[:,3]

testing_data = testing.ix[:,0:2]

model = DecisionTreeClassifier()
model.fit(training_data, training_target)

# make predictions
#expected = training_target
predicted = model.predict(testing_data)
   
# summarize the fit of the model
#print("\nThe model info : \n")
#print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))
   
#print("The accuracy: ")
#print(metrics.accuracy_score(expected,predicted))

id = pd.DataFrame(list(range(1, len(predicted) + 1)),columns=['Id'])


predicted = pd.DataFrame(predicted,columns=['Prediction'])
output = pd.concat([id, predicted], axis=1)

pd.DataFrame(output).to_csv('D:/prediction.csv',index=False)


