from rulekit.experiment import ExperimentRunner
from rulekit import RuleKit
import sys

def main():
    rulekit = RuleKit()
    rulekit.init()
    ExperimentRunner.run(sys.argv[1:])

main()