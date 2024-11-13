from sklearn.datasets import load_iris

from rulekit.classification import RuleClassifier

X, y = load_iris(return_X_y=True)

clf = RuleClassifier()
clf.fit(X, y)
prediction = clf.predict(X)

print(prediction)
