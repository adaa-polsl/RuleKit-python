from xml.etree import ElementTree
import pandas as pd
from scipy.io.arff import loadarff
from sklearn import metrics
from jpype import JClass
from typing import List, Dict, Tuple, Union
import os
import re

from rulekit.helpers import create_example_set, _fix_missing_values
from rulekit.rules import Rule

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

TEST_CONFIG_PATH = f'{dir_path}/resources/config'
REPORTS_IN_DIRECTORY_PATH = f'{dir_path}/resources/reports'
DATA_IN_DIRECTORY_PATH = f'{dir_path}/resources/'
REPORTS_OUT_DIRECTORY_PATH = f'{dir_path}/test_out'

REPORTS_SECTIONS_HEADERS = {
    'RULES': 'RULES'
}

DATASETS_PATH = '/resources/data'
EXPERIMENTS_PATH = '/resources/config'


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

    for column in values:
        _fix_missing_values(values[column])
    _fix_missing_values(labels)
    return ExampleSetWrapper(values, labels)


def get_dataset_path(name: str) -> str:
    return f'{DATASETS_PATH}/{name}'


class Knowledge:

    def __init__(self):
        self.expert_rules = []
        self.expert_preferred_conditions = []
        self.expert_forbidden_conditions = []


class TestReport:

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.rules = None


class TestCase:

    def __init__(self):
        self.param_config: Dict[str, object] = None
        self._reference_report: TestReport = None
        self._example_set: ExampleSetWrapper = None
        self.induction_params: Dict = None
        self.knowledge: Knowledge = None

        self.name: str = None
        self.data_set_file_path: str = None
        self.label_attribute: str = None
        self.survival_time: str = None
        self.report_file_path: str = None
        self.using_existing_report_file: bool = False

    @property
    def example_set(self) -> ExampleSetWrapper:
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
        self.datasets: List[DataSetConfig] = None


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

    def _parse_survival_time(self, element) -> Union[str, None]:
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
        return expert_rules if len(expert_rules) > 0 else None

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
                value = self._parse_experts_rules_parameters(param_node.findall(TestConfigParser.ENTRY_KEY))
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
            test_case_name: str,
            params: Dict[str, object],
            data_set_config: DataSetConfig) -> TestCase:
        test_case = TestCase()
        self.fix_params_typing(params)
        test_case.induction_params = params
        test_case.data_set_file_path = f'{DATA_IN_DIRECTORY_PATH}/{data_set_config.train_file_name}'
        test_case.label_attribute = data_set_config.label_attribute
        test_case.name = test_case_name
        test_case.param_config = params
        return test_case

    def fix_params_typing(self, params: dict):
        for key, value in params.items():
            if value == 'false':
                params[key] = False
                continue
            if value == 'true':
                params[key] = True
                continue
            if not 'measure' in key:
                params[key] = int(float(value))

    def make(self, tests_configs: Dict[str, TestConfig], report_dir_path: str) -> List[TestCase]:
        test_cases = []
        for key in tests_configs.keys():
            test_config = tests_configs[key]
            for config_name in test_config.parameter_configs.keys():
                for data_set_config in test_config.datasets:
                    params = test_config.parameter_configs[config_name]
                    test_case_name = f'{key}.{config_name}.{data_set_config.name}'
                    test_config.parameter_configs[config_name].pop('use_expert', None)
                    expert_rules = test_config.parameter_configs[config_name].pop('expert_rules', None)
                    preferred_conditions = test_config.parameter_configs[config_name].pop('expert_preferred_conditions',
                                                                                          None)
                    forbidden_conditions = test_config.parameter_configs[config_name].pop('expert_forbidden_conditions',
                                                                                          None)
                    test_case = self._make_test_case(test_case_name,
                                                     test_config.parameter_configs[config_name], data_set_config)
                    if 'use_report' in params:
                        report_file_name = params['use_report']
                        test_case.using_existing_report_file = True
                    else:
                        report_file_name = test_case_name
                    report_path = f'{report_dir_path}/{report_file_name}'
                    test_case.report_file_path = report_path
                    test_case.survival_time = data_set_config.survival_time
                    if expert_rules is not None or preferred_conditions is not None or forbidden_conditions is not None:
                        test_case.knowledge = Knowledge()
                        test_case.knowledge.expert_rules = expert_rules
                        test_case.knowledge.expert_forbidden_conditions = forbidden_conditions
                        test_case.knowledge.expert_preferred_conditions = preferred_conditions
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
    if not os.path.exists(DATA_IN_DIRECTORY_PATH):
        raise Exception('''\n
Test resources directory dosen't exist. Check if 'tests/resources/' directory exist.

If you're running tests for the first time you need to download resources folder from RuleKit repository by running:
    python tests/resources.py download
        ''')
    configs = TestConfigParser().parse(f'{TEST_CONFIG_PATH}/{class_name}.xml')
    return TestCaseFactory().make(configs, f'{REPORTS_IN_DIRECTORY_PATH}/{class_name}/')


def _get_rule_string(rule: Rule) -> str:
    return re.sub(r'(\\[[^\\]]*\\]$)|(\\([^\\)]*\\)$)', '', str(rule))


def assert_rules_are_equals(expected: List[str], actual: List[str]):
    def sanitize_rule_string(rule_string: str) -> str:
        return re.sub(r'(\t)|(\n)|(\[[^\]]*\]$)', '', rule_string)

    if len(expected) != len(actual):
        raise AssertionError(
            f'Rulesets have different number of rules, actual: {len(actual)}, expected: {len(expected)}')
    dictionary = {}
    for rule in expected:
        dictionary[sanitize_rule_string(rule)] = 0
    for rule in actual:
        key = sanitize_rule_string(rule)
        if key in dictionary:
            dictionary[key] = dictionary[key] + 1
        else:
            pass
            raise AssertionError('Actual ruleset contains rules not present in expected ruleset')
    for key in dictionary.keys():
        if dictionary[key] == 0:
            raise AssertionError('Ruleset are not equal, some rules are missing')
        elif dictionary[key] > 1:
            raise AssertionError('Somes rules were duplicated')


def assert_accuracy_is_greater(prediction, expected, threshold: float):
    labels = expected.to_numpy().astype(str)
    acc = metrics.accuracy_score(labels, prediction)
    if acc <= threshold:
        raise AssertionError(f'Accuracy should be greater than {threshold}')


def assert_score_is_greater(prediction, expected, threshold: float):
    if isinstance(prediction[0], int):
        labels = expected.to_numpy().astype(int)
    if isinstance(prediction[0], float):
        labels = expected.to_numpy().astype(float)
    explained_variance_score = metrics.explained_variance_score(labels, prediction)

    if explained_variance_score <= threshold:
        raise AssertionError(f'Score should be greater than {threshold}')
