"""Module defining CLI
"""
import os
import sys
import warnings

from rulekit import RuleKit
from rulekit._experiment import _ExperimentRunner

dir_path = os.path.dirname(os.path.realpath(__file__))


def _main():
    if len(sys.argv) > 1 and sys.argv[1] == 'download_jar':
        warnings.warn(
            (
                'This command is deprecated. From major version 2, '
                'RuleKit jar file is already packed with this package '
                'distribution and there is no need to download it in '
                'additional step. \n\n Currently this command will do '
                'nothing. It will be completely removed in the next '
                'major version.'
            ),
            DeprecationWarning,
            stacklevel=2
        )
        return
    else:
        # use rulekit batch CLI
        rulekit = RuleKit()
        rulekit.init()
        _ExperimentRunner.run(sys.argv[1:])


if __name__ == "__main__":
    _main()
