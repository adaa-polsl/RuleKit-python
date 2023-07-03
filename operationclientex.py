from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from sklearn.datasets import load_iris

#Initialize Rulekit (Java connection is also initialized)
RuleKit.init()

#Implementation of Java interface to receive progress and rule events and set stop of processing
from jpype import JImplements, JOverride
from adaa.analytics.rules.operator import ICommandProxyClient

@JImplements(ICommandProxyClient)
class ExampleCommandProxyClient:
        
    @JOverride
    def onNewRule(self, rule):
        print("Rule",rule,"\n")
    
    @JOverride
    def onProgress(self, totalRules, uncoveredRules):
        print("Progress total: ",totalRules," uncovered: ",uncoveredRules,"\n")
      
    @JOverride
    def isRequestStop(self) -> bool:
        return False


#Run classification        
x, y = load_iris(return_X_y=True) 
clf = RuleClassifier()

#Add class implementing interface as interceptor to receive events
clf.add_operator_command_proxy(ExampleCommandProxyClient())

clf.fit(x, y)
prediction = clf.predict(x)
print(prediction)