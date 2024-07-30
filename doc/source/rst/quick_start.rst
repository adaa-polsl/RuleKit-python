
Quick start
========================================

.. warning::
   This package is python wrapper for java library called RuleKit. This means it **requires** java JRE in version **1.8.0** to
   be installed to run. You also need *JAVA_HOME* environmental variable to be set. Package should work fine with both Oracle and
   Open JDK.

Installation
-------------

.. code-block:: bash

    pip install rulekit

.. note::

To check if everything was installed correctly call:

.. code-block:: python

    import rulekit
    rulekit.__version__

It should run without errors and print package version.

Package usage
--------------------

Now we are finally ready to use rulekit package and its models.

.. code-block:: python

    from  sklearn import  datasets
    from rulekit import RuleKit
    from rulekit.classification import RuleClassifier

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    classifier = RuleClassifier()
    classifier.fit(X, y)

    prediction = classifier.predict(X)

    from sklearn.metrics import accuracy_score

    print('Accuracy: ', accuracy_score(y, prediction))
    

As you may noticed, training and usage of rulekit models is the same as in scikit learn. This 
mean you easily can use scikit: metrics, cross-validation, hyper-parameters tuning etc.


For more examples head to :doc:`Tutorials <./tutorials>` section.
