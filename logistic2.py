#importing the usual suspects
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("D:\\MCA\\MCA 5 SEM\\ml\\pro\\logistic\\HR_comma_sep.csv")
df.head()
#dispalying content
df.describe()
# to  remove NUll/cleaning
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis')
#just to make sure
df[df.isnull()].count()





#fitting the model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
X = df[['satisfaction_level','number_project',
       'sal_class','Work_accident']]
y = df['left']

sal_class=pd.get_dummies(X['sal_class'],drop_first=True)

# Drop the state coulmn
X=X.drop('sal_class',axis=1)

# concat the dummy variables
X=pd.concat([X,sal_class],axis=1)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


from sklearn.metrics import r2_score
score=r2_score(y_test,y_train)
