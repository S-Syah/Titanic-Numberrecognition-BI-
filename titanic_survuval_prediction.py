from google.colab import files
uploaded=files.upload()      #import the required dataset in google colab
import pandas as p
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
data = p.read_csv('titanic.csv')
data.shape
data.info
data.head()
d=p.get_dummies(data.Embarked,prefix="Embarked")
d.head()
data=data.join(d)
data.drop(columns=['Embarked'],axis=1,inplace=True)
data.Sex=data.Sex.map({"male":1,'female':0})
data.Sex
y=data.Survived.copy()
x=data.drop(["Survived"],axis=1)
x.drop(['Cabin','Ticket','Name','PassengerId'],axis=1,inplace=True)
x.info
x.Age.fillna(x.Age.mean(),inplace=True)
x.isnull().values.any()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)
sns.countplot(x='Survived', hue='Sex', data=data)
plt.show()
