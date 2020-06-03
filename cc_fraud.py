#The downloaded archive is named 310_23498_bundle_archive.zip and contains a single csv file: creditcard.csv

# import all packages
from zipfile import ZipFile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix


# unzip the downloaded archive
with ZipFile('310_23498_bundle_archive.zip', 'r') as zipObj:
   zipObj.extractall()

# Read the creditcard.csv file into a pandas dataframe:
df = pd.read_csv('creditcard.csv')

# Check the number of rows and columns. Should be 284,807 rows, one for each transaction and 31 columns.
# Time, V1 to V28 and Amount are the feature columns 
# Class column is the indicator of non-fraudulent (Class=0) or fraudulent (Class=1) transactions.
print('The dataframe has',df.shape[0],'rows and',df.shape[1],'columns.')

# Check for missing values
print('There are',df.isnull().sum().sum(),'missing values.')


# Check class balance
class_0 = df['Class'].value_counts()[0]
class_1 = df['Class'].value_counts()[1]
print('There are',class_0,'non-fraudulent transactions and',class_1,'fraudulent transactions.')


# Training the Random Forest Classifier
# First, we need to separate the dataframe into features (X) and class labels (y):
X = df.iloc[:,:30]
y = df.iloc[:,30]

# Now use SciKit-Learn's train-test split utility to divide X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# train the RF classifier
print ('Starting training..')
crf=RandomForestClassifier(n_estimators=200)
crf.fit(X_train,y_train)

# Now make predictions using the test set
predictions=crf.predict(X_test)

n_errors = (predictions != y_test).sum()
print('Errors:', n_errors)

acc= accuracy_score(y_test,predictions)
print("The accuracy is  {}".format(acc))
prec= precision_score(y_test,predictions)
print("The precision is {}".format(prec))
rec= recall_score(y_test,predictions)
print("The recall is {}".format(rec))
f1= f1_score(y_test,predictions)
print("The F1-Score is {}".format(f1))
MCC=matthews_corrcoef(y_test,predictions)
print("The Matthews correlation coefficient is {}".format(MCC))

