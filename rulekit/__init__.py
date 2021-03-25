
import os
from .main import RuleKit

current_path = os.path.dirname(os.path.realpath(__file__))

__version__ = open(f'{current_path}/VERSION.txt', mode='r').read().strip()

RuleKit = RuleKit