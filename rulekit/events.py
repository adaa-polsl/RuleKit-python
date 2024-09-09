"""Module containing classes for tracking progress of rule induction.
"""
from typing import Any

from jpype import JImplements
from jpype import JOverride

from rulekit.rules import Rule


class RuleInductionProgressListener:
    """Base class for rule induction progress listeners. To use it, subclass it
    and implement some of the its methods. Then instantiate it and pass it to
     `add_event_listener` method of the operator.
    """

    def on_new_rule(self, rule: Rule):
        """Called when new rule is induced

        Args:
            rule (Rule): Newly induced rule
        """

    def on_progress(
        self,
        total_examples_count: int,
        uncovered_examples_count: int
    ):
        """Called each time a ruleset coverage changed.

        This method is best suited to monitor progress of rule induction.

        Args:
            total_examples_count (int): Total number of examples in training dataset
            uncovered_examples_count (int): Number of examples that are not covered by any rule
        """

    def should_stop(self) -> bool:
        """Method which allows to stop rule induction process at given moment. This method is
         called each time a ruleset coverage changed. If it returns `True`, rule induction process
        will be stopped if it return `False` it will continue.

        Returns:
            bool: whether to stop rule induction or not
        """
        return False


def command_listener_factory(listener: RuleInductionProgressListener) -> Any:
    from adaa.analytics.rules.logic.rulegenerator import \
        ICommandListener  # pylint: disable=import-outside-toplevel,import-error

    @JImplements(ICommandListener)
    class _CommandListener:  # pylint: disable=invalid-name,missing-function-docstring

        @JOverride
        def onNewRule(self, rule):
            return listener.on_new_rule(Rule(rule))

        @JOverride
        def onProgress(self, totalRules: int, uncoveredRules: int):
            return listener.on_progress(totalRules, uncoveredRules)

        @JOverride
        def isRequestStop(self) -> bool:
            return listener.should_stop()
            # return False

    return _CommandListener()
