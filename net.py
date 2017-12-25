


#Download the dataset from link below
#https://he-s3.s3.amazonaws.com/media/hackathon/machine-learning-challenge-4/sample/c771afa0-c-HackerEarthML4Updated.zip

import numpy as np
from sklearn import cross_validation
import pandas as pd
from pandas import Series,DataFrame
from sklearn.ensemble import AdaBoostClassifier

df=pd.read_csv('train_data.csv')
df.drop(['connection_id'],1,inplace=True)


x=np.array(df.drop(['target'],1))
y=np.array(df['target'])

xtr,xt,ytr,yt=cross_validation.train_test_split(x,y,test_size=0.2)
clf=AdaBoostClassifier(n_estimators=42)
clf.fit(xtr,ytr)
#score=cross_val_score(clf,x,y)
confidence=clf.score(xt,yt)
print(confidence)
df1=pd.read_csv('test_data.csv')
df2=df1
df1=df1.drop(['connection_id'],1)
xts=np.array(df1)
preds=np.array(clf.predict(xts))
df2['target']=pd.Series(preds)
df2=DataFrame(data=df2,columns=['connection_id','target'])
df2.to_csv('predicts.csv',index=False)



	


