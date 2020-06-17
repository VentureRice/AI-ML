import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import plot_importance
from mlxtend.classifier import StackingClassifier
import argparse
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

# 参数
parser = argparse.ArgumentParser(description='Choose Classifier.')
parser.add_argument('classifier', default = 'xgboost',type = str,
                    help='choose a classifier: logistic, xgboost, randomforest, svm, adaboost, gbdtlr')
parser.add_argument('test_size', default = 0.4,type = float,
                    help='the size of test dataset')

args = parser.parse_args()
classifier = args.classifier
test_size = args.test_size

# 模型评估
def plot_roc(roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('./'+classifier+'.jpg',dpi=500,bbox_inches = 'tight')

# 读取数据
status_res = pd.read_csv('status_res.csv')
status_res = status_res.fillna(0)
cols = list(status_res.columns)
status_res.iloc[:,3:] = abs(status_res.iloc[:,3:])
for i in range(3,len(cols)):
    status_res[cols[i]].replace(np.inf,max(status_res[cols[i]].replace(np.inf,0,inplace=False)),inplace=True)
# 划分数据集
X = status_res.iloc[:,3:]
del X['if_bad']
y = status_res['if_bad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

if classifier=='xgboost':
	xgb = XGBClassifier(max_depth=8)
	xgb.fit(X_train,y_train)
	pred = xgb.predict_proba(X_test)
	
elif classifier=='logistic':
	lr = LogisticRegression()
	lr.fit(X_train,y_train)
	pred = lr.predict_proba(X_test)

elif classifier=='svm':
	svm_model = svm.SVC(probability=True)
	svm_model.fit(X_train,y_train)
	pred = svm_model.predict_proba(X_test)

elif classifier=='randomforest':
	rfc = RandomForestClassifier()
	rfc.fit(X_train,y_train)
	pred = rfc.predict_proba(X_test)

elif classifier=='adaboost':
	adb = AdaBoostClassifier()
	adb.fit(X_train,y_train)
	pred = adb.predict_proba(X_test)
elif classifier=='stacking':
	clf1 = AdaBoostClassifier()
	clf2 = RandomForestClassifier(random_state=1)
	clf3 = XGBClassifier(max_depth=10)
	clf4 = svm.SVC(probability=True)
	lr = LogisticRegression()
	sclf = StackingClassifier(classifiers=[clf3, clf2, clf4], 
	                          meta_classifier=lr)
	sclf.fit(X_train,y_train)
	pred = sclf.predict_proba(X_test)
elif classifier=='gbdtlr':
	gbm1 = GradientBoostingClassifier(n_estimators=500, max_depth=55)
	gbm1.fit(X_train, y_train)
	train_new_feature = gbm1.apply(X_train)

	train_new_feature = train_new_feature.reshape(-1, train_new_feature.shape[1])
	enc = OneHotEncoder()
	enc.fit(train_new_feature)
	test_new_feature = gbm1.apply(X_test)
	test_new_feature = test_new_feature.reshape(-1, test_new_feature.shape[1])
	test_new_feature = np.array(enc.transform(test_new_feature).toarray())
	train_new_feature2 = np.array(enc.transform(train_new_feature).toarray())
	lr = LogisticRegression()
	lr.fit(train_new_feature2,y_train)
	pred = lr.predict_proba(test_new_feature)

else:
	print('choose a classifier: logistic, xgboost, randomforest, svm, adaboost, stacking')


fpr,tpr,_ = roc_curve(y_test, pred[:, 1])
roc_auc = auc(fpr, tpr)

ratio = sum(y)/len(y)
pred = pred[:, 1]
for i in range(len(pred)):
	if pred[i]>ratio:
		pred[i]=1
	else:
		pred[i]=0

acc = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred)
pre = precision_score(y_test, pred) 
f1 = f1_score(y_test, pred) 
print('auc =',round(roc_auc,4))
print('accracy =',round(acc,4))
print('recall =',round(recall,4))
print('precision =',round(pre,4))
print('F1 score = ', round(f1,4))
plot_roc(roc_auc)




