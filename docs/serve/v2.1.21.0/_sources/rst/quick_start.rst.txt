
Quick start
========================================

.. warning::
    This package is a wrapper for Java library, and requires Java Development Kit version 8 
    or later to be installed on the computer. Both Open JDK and Oracle implementations are supported.

    If you don't have JDK installed on your computer you can quickly set it up using 
    :ref:`install-jdk<https://pypi.org/project/install-jdk/>` package.

    .. code-block:: bash
        
        pip install install-jdk

    .. code-block:: python

        import jdk

        jdk.install('11', jre=True)

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
