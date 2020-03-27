import os

import jpype
import jpype.imports
from jpype import JClass
from jpype.types import *

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
JAR_FILENAME = 'rulekit-1.1.0-all.jar'
PATH_TO_JAR = f'{CURRENT_PATH}\\..\\..\\adaa.analytics.rules\\build\\libs\\{JAR_FILENAME}'


if __name__ == "__main__":
  # Launch the JVM
  jpype.startJVM(jpype.getDefaultJVMPath(),"-Djava.class.path=%s"%PATH_TO_JAR, convertStrings=False)


  Interval = JClass('adaa.analytics.rules.logic.representation.Interval')

  print(Interval)

  interval = Interval()
  print(interval.equals(interval))



