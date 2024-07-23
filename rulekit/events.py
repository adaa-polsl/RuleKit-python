from typing import Any

from jpype import JImplements, JOverride

from .rules import Rule


class RuleInductionProgressListener:

    def on_new_rule(self, rule: str):
        pass

    def on_progress(
        self,
        total_examples_count: int,
        uncovered_examples_count: int
    ):
        pass

    def should_stop(self) -> bool:
        return False


def command_listener_factory(listener: RuleInductionProgressListener) -> Any:
    from adaa.analytics.rules.logic.rulegenerator import \
        ICommandListener  # pylint: disable=import-outside-toplevel

    @JImplements(ICommandListener)
    class CommandListener:

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

    return CommandListener()
