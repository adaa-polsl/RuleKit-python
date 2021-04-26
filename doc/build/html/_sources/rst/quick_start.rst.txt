
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
    python -m rulekit download_jar

.. note::
   Second command will download `RuleKit <https://github.com/adaa-polsl/RuleKit/>`__ jar
   file from github releases. This step is required to use this package.

To check if everything was installed correctly call:

.. code-block:: python

    import rulekit
    rulekit.__version__

It should run without errors and print package version.

Initializing package
--------------------

Before you start using any of rulelkit package functionality you need to initialize it first. This step should be done only
once at the beginning of the program, no need to initialize before every usage.

.. code-block:: python

    from rulekit import RuleKit

    RuleKit.init()
  
    print(RuleKit.version)

If this step failed it probably means one of two things:

- you do not have java installed on your computer. Run :code:`'java -version'` and check for error and JRE version (it should be 1.8.0)

- there is no *'rulekit-*-all.jar'* file in *'jar'* directory of the package. You can get jar file from `here <https://github.com/adaa-polsl/RuleKit/releases>`_ (download file ending with *'-all.jar'*).

If everything worked fine it should print RuleKit jar version on the screen. You may wonder what is the difference between this version and the one printed at the beginning of this section. The first one is a version of python wrapper itself whereas the second on is a version of the `RuleKit <https://github.com/adaa-polsl/RuleKit>`_ library that is being used by the wrapper.

Package usage
--------------------

Now we are finally ready to use rulekit package and its models.

.. code-block:: python

    from  sklearn import  datasets
    from rulekit import RuleKit
    from rulekit.classification import RuleClassifier

    iris=datasets.load_iris()
    X=iris.data
    y=iris.target

    # don't forget to call init!
    RuleKit.init()

    classifier = RuleClassifier()
    classifier.fit(X, y)

    prediction = classifier.predict(X)

    from sklearn.metrics import accuracy_score

    print('Accuracy: ', accuracy_score(y, prediction))
    

As you may noticed, training and using rulekit models is the same as in scikit learn. This 
mean you can use scikit: metrics, cross-validation, hyper-parameters tuning etc. with ease. 


For more examples head to :doc:`Tutorials <./tutorials>` section.
