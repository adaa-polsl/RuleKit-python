from xml.etree import ElementTree
import pandas as pd
from scipy.io.arff import loadarff
from jpype import JClass, JString, JObject, JArray, JInt
from typing import List, Dict, Tuple
import os
import sys
import re

from rulekit.tree.helpers import create_example_set, set_survival_time
from rulekit.tree.rules import Rule

TEST_CONFIG_PATH = '../adaa.analytics.rules/test/resources/config'
REPORTS_IN_DIRECTORY_PATH = '../adaa.analytics.rules/test/resources/reports'
DATA_IN_DIRECTORY_PATH = '../adaa.analytics.rules/test/resources/'
REPORTS_OUT_DIRECTORY_PATH = '../test_out'

REPORTS_SECTIONS_HEADERS = {
    'RULES': 'RULES'
}

DATASETS_PATH = '../data'
EXPERIMENTS_PATH = '../adaa.analytics.rules/test/resources/config'


class ExampleSetWrapper:

    def __init__(self, values, labels):
        self.values = values
        self.labels = labels
        self.example_set = create_example_set(values, labels)

    def get_data(self) -> Tuple:
        return self.values, self.labels


def load_arff_to_example_set(path: str, label_attribute: str) -> ExampleSetWrapper:
    arff_data_frame = pd.DataFrame(loadarff(path)[0])

    attributes_names = []
    for column_name in arff_data_frame.columns:
        if column_name != label_attribute:
            attributes_names.append(column_name)

    values = arff_data_frame[attributes_names]
    labels = arff_data_frame[label_attribute]
    return ExampleSetWrapper(values, labels)


def get_dataset_path(name: str) -> str:
    return f'{DATASETS_PATH}/{name}'


class KnowledgeConfigurator:

    @staticmethod
    def configure(knowledge, params: Dict):
        for key in params.keys():
            if key == 'extend_using_preferred':
                knowledge.setExtendUsingPreferred(params[key] == 'true')
            if key == 'extend_using_automatic':
                knowledge.setExtendUsingAutomatic(params[key] == 'true')
            if key == 'induce_using_preferred':
                knowledge.setInduceUsingPreferred(params[key] == 'true')
            if key == 'induce_using_automatic':
                knowledge.setInduceUsingAutomatic(params[key] == 'true')
            if key == 'consider_other_classes':
                knowledge.setConsiderOtherClasses(params[key] == 'true')
            if key == 'preferred_conditions_per_rule':
                knowledge.setPreferredConditionsPerRule(JInt(params[key]))
            if key == 'preferred_attributes_per_rule':
                knowledge.setPreferredAttributesPerRule(JInt(params[key]))


class KnowledgeFactory:

    PARAMETER_EXPERT_RULES = 'expert_rules'
    PARAMETER_EXPERT_PREFERRED_CONDITIONS = 'expert_preferred_conditions'
    PARAMETER_EXPERT_FORBIDDEN_CONDITIONS = 'expert_forbidden_conditions'

    def __init__(self, example_set):
        ExampleSetMetaData = JClass('com.rapidminer.operator.ports.metadata.ExampleSetMetaData')
        self._example_set = example_set
        self._example_set_meta_data = ExampleSetMetaData(example_set)

    def _fix_mappings(self, rules, example_set):
        ElementaryCondition = JClass('adaa.analytics.rules.logic.representation.ElementaryCondition')
        Attribute = JClass('adaa.analytics.rules.logic.induction.Attribute')
        SingletonSet = JClass('adaa.analytics.rules.logic.representation.SingletonSet')
        for rule in rules:
            for condition_base in rule.getPremise().getSubconditions():
                if isinstance(condition_base, ElementaryCondition):
                    elementary_condition = JObject(condition_base, ElementaryCondition)
                else:
                    elementary_condition = None
                if elementary_condition is not None:
                    attribute = example_set.getAttributes().get(elementary_condition.getAttribute())
                    if attribute.isNominal() and isinstance(elementary_condition.getValueSet(), SingletonSet):
                        singleton_set = JObject(elementary_condition, SingletonSet)
                        value_name = singleton_set.getMapping().get(JInt(singleton_set.getValue()))
                        new_value = attribute.getMapping().getIndex(value_name)
                        singleton_set.setValue(new_value)
                        singleton_set.setMapping(attribute.getMapping().getValues())

    def _make_experts_rules(self, elements: List[str]) -> object:
        MultiSet = JClass('adaa.analytics.rules.logic.representation.MultiSet')
        RuleParser = JClass('adaa.analytics.rules.logic.representation.RuleParser')
        ConditionBase = JClass('adaa.analytics.rules.logic.representation.ConditionBase')
        rules = MultiSet()
        for element in elements:
            rule = RuleParser.parseRule(element[1], self._example_set_meta_data)
            if rule is not None:
                for condition_base in rule.getPremise().getSubconditions():
                    condition_base.setType(ConditionBase.Type.FORCED)
                rules.add(rule)
        return rules

    def _make_preferred_conditions(self, elements: List[str]) -> object:
        RuleParser = JClass('adaa.analytics.rules.logic.representation.RuleParser')
        MultiSet = JClass('adaa.analytics.rules.logic.representation.MultiSet')
        ConditionBase = JClass('adaa.analytics.rules.logic.representation.ConditionBase')
        preferred_conditions = MultiSet()
        pattern = r'(?<number>(\\d+)|(inf)):\\s*(?<rule>.*)'
        for element in elements:
            match = re.search(pattern, element)
            count = match.group('number')
            rule_desc = match.group('rule')
            rule = RuleParser.parseRule(rule_desc, self._example_set_meta_data)
            if rule is not None:
                rule.getPremise().setType(ConditionBase.Type.PREFERRED)
                for cnd in rule.getPremise().getSubconditions():
                    cnd.setType(ConditionBase.Type.NORMAL)
                if count == 'inf':
                    count = sys.maxsize
                else:
                    count = int(count)
                count = JInt(count)
                preferred_conditions.add(rule, count)
        return preferred_conditions

    def _make_forbidden_conditions(self, elements: List[str]) -> object:
        RuleParser = JClass('adaa.analytics.rules.logic.representation.RuleParser')
        MultiSet = JClass('adaa.analytics.rules.logic.representation.MultiSet')
        ConditionBase = JClass('adaa.analytics.rules.logic.representation.ConditionBase')
        forbidden_conditions = MultiSet()
        for element in elements:
            rule = RuleParser.parseRule(element[1], self._example_set_meta_data)
            for cnd in rule.getPremise().getSubconditions():
                cnd.setType(ConditionBase.Type.NORMAL)
            forbidden_conditions.add(rule)
        return forbidden_conditions

    def make(self, params: Dict[str, object]) -> object:
        MultiSet = JClass('adaa.analytics.rules.logic.representation.MultiSet')
        Rule = JClass('adaa.analytics.rules.logic.representation.Rule')
        Knowledge = JClass('adaa.analytics.rules.logic.representation.Knowledge')

        rules = MultiSet()
        preferred_conditions = MultiSet()
        forbidden_conditions = MultiSet()

        for key in params.keys():
            if key == KnowledgeFactory.PARAMETER_EXPERT_RULES:
                rules = self._make_experts_rules(params[key])
            if key == KnowledgeFactory.PARAMETER_EXPERT_PREFERRED_CONDITIONS:
                rules = self._make_preferred_conditions(params[key])
            if key == KnowledgeFactory.PARAMETER_EXPERT_FORBIDDEN_CONDITIONS:
                rules = self._make_forbidden_conditions(params[key])
        self._fix_mappings(rules, self._example_set)
        self._fix_mappings(preferred_conditions, self._example_set)
        self._fix_mappings(forbidden_conditions, self._example_set)

        knowledge = Knowledge(self._example_set, rules, preferred_conditions, forbidden_conditions)
        KnowledgeConfigurator.configure(params)
        return knowledge


class TestReport:

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.rules = None


class TestCase:

    def __init__(self):
        self.param_config: Dict[str, object] = None
        self._reference_report: TestReport = None
        self._example_set: ExampleSetWrapper = None
        self.induction_params = None
        self.knowledge = None

        self.name: str = None
        self.data_set_file_path: str = None
        self.label_attribute: str = None
        self.survival_time: str = None
        self.report_file_path: str = None
        self.using_existing_report_file: bool = False

    @property
    def example_set(self) -> object:
        if self._example_set is None:
            self._example_set = load_arff_to_example_set(self.data_set_file_path, self.label_attribute)
        return self._example_set

    @property
    def reference_report(self) -> TestReport:
        if self._reference_report is None:
            ExampleSetMetaData = JClass('com.rapidminer.operator.ports.metadata.ExampleSetMetaData')
            reader = TestReportReader(f'{self.report_file_path}.txt', ExampleSetMetaData(self._example_set.example_set))
            self._reference_report = reader.read()
            reader.close()
        return self._reference_report


class DataSetConfig:

    def __init__(self):
        self.name: str = None
        self.label_attribute: str = None
        self.train_file_name: str = None
        self.survival_time: str = None


class TestConfig:

    def __init__(self):
        self.name: str = None
        self.parameter_configs: Dict[str, Dict[str, object]] = None
        self.datasets: str = List[DataSetConfig]


class TestConfigParser:
    TEST_KEY = 'test'
    NAME_KEY = 'name'
    IN_FILE_KEY = 'in_file'
    TRAINING_KEY = 'training'
    TRAIN_KEY = 'train'
    LABEL_KEY = 'label'
    DATASET_KEY = 'dataset'
    DATASETS_KEY = 'datasets'
    PARAM_KEY = 'param'
    PARAMETERS_SET_KEY = 'parameter_sets'
    PARAMETERS_KEY = 'parameter_set'
    ENTRY_KEY = 'entry'
    SURVIVAL_TIME_ROLE = 'survival_time'

    EXPERTS_RULES_PARAMETERS_NAMES = [
        'expert_rules',
        'expert_preferred_conditions',
        'expert_forbidden_conditions'
    ]

    def __init__(self):
        self.tests_configs: Dict[str, TestConfig] = {}
        self.root: ElementTree = None

    def _parse_survival_time(self, element) -> str:
        survival_time_element = element.find(TestConfigParser.SURVIVAL_TIME_ROLE)
        if element.find(TestConfigParser.SURVIVAL_TIME_ROLE) is not None:
            return survival_time_element.text
        else:
            return None

    def _parse_experts_rules_parameters(self, elements) -> List[Tuple]:
        expert_rules = []
        for element in elements:
            rule_name = element.attrib['name']
            rule_content = element.text
            expert_rules.append((rule_name, rule_content))
        return expert_rules

    def _check_ambigous_data_sets_names(self, data_sets_configs: List[DataSetConfig]):
        dict = {}
        for element in data_sets_configs:
            dict[element.name] = None
        if len(dict.keys()) < len(data_sets_configs):
            raise ValueError('Datasets are ambigous')

    def _parse_data_set(self, element) -> DataSetConfig:
        data_set_config = DataSetConfig()
        data_set_config.label_attribute = element.find(TestConfigParser.LABEL_KEY).text
        train_element = element.find(TestConfigParser.TRAINING_KEY)
        train_element = train_element.find(TestConfigParser.TRAIN_KEY)
        data_set_config.train_file_name = train_element.find(TestConfigParser.IN_FILE_KEY).text
        data_set_config.name = element.attrib.get('name', None)
        if data_set_config.name is None:
            file_name = os.path.basename(data_set_config.train_file_name)
            data_set_config.name = file_name.split('.')[0]
        data_set_config.survival_time = self._parse_survival_time(element)
        return data_set_config

    def _parse_data_sets(self, element) -> List[DataSetConfig]:
        data_set_configs = []
        node = element.find(TestConfigParser.DATASETS_KEY)
        for element in node.findall(TestConfigParser.DATASET_KEY):
            data_set_configs.append(self._parse_data_set(element))
        return data_set_configs

    def parse_test_parameters(self, element) -> Dict[str, object]:
        params = {}
        for param_node in element.findall(TestConfigParser.PARAM_KEY):
            name: str = param_node.attrib['name']
            if name in TestConfigParser.EXPERTS_RULES_PARAMETERS_NAMES:
                value = self._parse_experts_rules_parameters(element.findall(TestConfigParser.ENTRY_KEY))
            else:
                value = param_node.text
            params[name] = value
        return params

    def _parse_test_parameters_sets(self, element) -> Dict[str, Dict[str, object]]:
        parameters_sets = {}
        params_sets_node = element.findall(TestConfigParser.PARAMETERS_SET_KEY)[0]
        for param_set in params_sets_node.findall(TestConfigParser.PARAMETERS_KEY):
            name: str = param_set.attrib['name']
            parameters_sets[name] = self.parse_test_parameters(param_set)
        return parameters_sets

    def _parse_test(self, element) -> TestConfig:
        test_config = TestConfig()
        test_config.parameter_configs = self._parse_test_parameters_sets(element)
        test_config.datasets = self._parse_data_sets(element)
        test_config.name = element.attrib['name']
        return test_config

    def parse(self, file_path: str) -> Dict[str, TestConfig]:
        self.tests_configs = {}
        self.root = ElementTree.parse(file_path).getroot()
        if self.root.tag == 'test':
            test_elements = [self.root]
        else:
            test_elements = self.root.findall(TestConfigParser.TEST_KEY)
        for test_element in test_elements:
            test_config = self._parse_test(test_element)
            self.tests_configs[test_config.name] = test_config
        return self.tests_configs


class TestCaseFactory:

    def _make_test_case(
            self,
            test_config:
            TestConfig,
            test_case_name: str,
            params: Dict[str, object],
            data_set_config: DataSetConfig) -> TestCase:
        test_case = TestCase()
        test_case.induction_params = params
        test_case.data_set_file_path = f'{DATA_IN_DIRECTORY_PATH}/{data_set_config.train_file_name}'
        test_case.label_attribute = data_set_config.label_attribute
        test_case.name = test_case_name
        test_case.param_config = params
        return test_case

    def make(self, tests_configs: Dict[str, TestConfig], report_dir_path: str) -> List[TestCase]:
        test_cases = []
        for key in tests_configs.keys():
            test_config = tests_configs[key]
            for config_name in test_config.parameter_configs.keys():
                for data_set_config in test_config.datasets:
                    params = test_config.parameter_configs[config_name]
                    test_case_name = f'{key}.{config_name}.{data_set_config.name}'
                    test_case = self._make_test_case(test_config, test_case_name, test_config.parameter_configs[config_name], data_set_config)
                    if 'use_report' in params:
                        report_file_name = params['use_report']
                        test_case.using_existing_report_file = True
                    else:
                        report_file_name = test_case_name
                    report_path = f'{report_dir_path}/{report_file_name}'
                    test_case.report_file_path = report_path
                    test_case.survival_time = data_set_config.survival_time
                    test_cases.append(test_case)
        return test_cases


def get_rule_string(rule) -> str:
    return re.sub(r'(\\[[^\\]]*\\]$)|(\\([^\\)]*\\)$)', '', str(rule))


class TestReportReader:

    def __init__(self, file_name: str, example_set_meta_data):
        self.example_set_meta_data = example_set_meta_data
        self.file_name = file_name
        self._file = f = open(file_name, "r")

    def _read_rules(self, test_report: TestReport):
        rules = []
        for line in self._file:
            if len(line) == 0:
                break
            else:
                rules.append(line)
        test_report.rules = rules

    def read(self) -> TestReport:
        test_report = TestReport(self.file_name)
        for line in self._file:
            line = line.upper()
            line = re.sub(r'\t', '', line)
            line = line.replace('\n', '')
            if line == REPORTS_SECTIONS_HEADERS['RULES']:
                self._read_rules(test_report)
            elif line == '':
                continue
            else:
                raise ValueError(f'Invalid report file format for file: {self.file_name}')
        return test_report

    def close(self):
        self._file.close()


class TestReportWriter:

    def __init__(self, file_name: str):
        self._file = f = open(file_name, "w")
        if not os.path.exists(REPORTS_OUT_DIRECTORY_PATH):
            os.makedirs(REPORTS_OUT_DIRECTORY_PATH)

    def write(self, rule_set):
        self._file.write('\n')
        self._file.write(f'{REPORTS_SECTIONS_HEADERS["RULES"]}\n')
        for rule in rule_set.rules:
            self._file.write(f'\t{get_rule_string(rule)}')

    def close(self):
        self._file.close()


def get_test_cases(class_name: str) -> List[TestCase]:
    configs = TestConfigParser().parse(f'{TEST_CONFIG_PATH}/{class_name}.xml')
    return TestCaseFactory().make(configs, f'{REPORTS_IN_DIRECTORY_PATH}/{class_name}/')


def _get_rule_string(rule: Rule) -> str:
    return re.sub(r'(\\[[^\\]]*\\]$)|(\\([^\\)]*\\)$)', '', str(rule))


def assert_rules_are_equals(expected: List[str], actual: List[str]):
    def sanitize_rule_string(rule_string: str) -> str:
        return re.sub(r'(\t)|(\n)|(\[.*\])', '', rule_string)

    if len(expected) != len(actual):
        raise AssertionError(f'Rulesets have different number of rules, actual: {len(actual)}, expected: {len(expected)}')
    dictionary = {}
    for rule in expected:
        dictionary[sanitize_rule_string(rule)] = 0
    for rule in actual:
        key = sanitize_rule_string(rule)
        if key in dictionary:
            dictionary[key] = dictionary[key] + 1
        else:
            raise AssertionError('Actual ruleset contains rules not present in expected ruleset')
    for key in dictionary.keys():
        if dictionary[key] == 0:
            raise AssertionError('Ruleset are not equal, some rules are missing')
        elif dictionary[key] > 1:
            raise AssertionError('Somes rules were duplicated')
