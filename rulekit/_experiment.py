"""Module containing class for running experiments using legacy
Java RuleKit CLI. For more details see  `documentation \
<https://github.com/adaa-polsl/RuleKit/wiki/7-Library-API#71-running-an-experiment
"""
from jpype import JClass


class _ExperimentRunner:

    @staticmethod
    def run(args: list[str]):
        """Run experiment using core Java RuleKit CLI

        Args:
            args (list[str]): args
        """
        ExperimentalConsole = JClass(  # pylint: disable=invalid-name
            'adaa.analytics.rules.consoles.ExperimentalConsole'
        )
        ExperimentalConsole.main(args)
