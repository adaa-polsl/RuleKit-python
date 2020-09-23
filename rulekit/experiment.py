from jpype import JClass
from typing import List


class ExperimentRunner:

    @staticmethod
    def run(args: List[str]):
        ExperimentalConsole = JClass('adaa.analytics.rules.consoles.ExperimentalConsole')
        ExperimentalConsole.main(args)
