import unittest

from jpype import JClass

from rulekit.main import RuleKit


class TestRuleKitMainClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_reading_version(self):
        self.assertIsNotNone(RuleKit.version, 'Version should not be None')
        self.assertEqual(len(RuleKit.version.split('.')), 3, 'Version should have correct format')

    def test_loading_rulekit_class(self):
        example_rulekit_class = JClass('adaa.analytics.rules.logic.rulegenerator.RuleGenerator')
        self.assertIsNotNone(example_rulekit_class, 'Should load RuleKit classes')


if __name__ == '__main__':
    unittest.main()