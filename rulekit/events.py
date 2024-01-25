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


def command_proxy_client_factory(listener: RuleInductionProgressListener) -> Any:
    from adaa.analytics.rules.operator import ICommandProxyClient  # pylint: disable=import-outside-toplevel

    @JImplements(ICommandProxyClient)
    class CommandProxyClient:

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

    return CommandProxyClient()
