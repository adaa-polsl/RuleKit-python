import io
import os
import re
from io import StringIO
from typing import Union
from xml.etree import ElementTree

import pandas as pd
from sklearn import metrics

from rulekit._helpers import ExampleSetFactory
from rulekit._problem_types import ProblemType
from rulekit.arff import read_arff

dir_path = os.path.dirname(os.path.realpath(__file__))

TEST_CONFIG_PATH = f"{dir_path}/resources/config"
REPORTS_IN_DIRECTORY_PATH = f"{dir_path}/resources/reports"
DATA_IN_DIRECTORY_PATH = f"{dir_path}/resources/"
REPORTS_OUT_DIRECTORY_PATH = f"{dir_path}/test_out"

REPORTS_SECTIONS_HEADERS = {"RULES": "RULES"}

DATASETS_PATH = "/resources/data"
EXPERIMENTS_PATH = "/resources/config"


def _fix_missing_values(column) -> None:
    for i, value in enumerate(column.values):
        if value == b"?":
            column.values[i] = None


class ExampleSetWrapper:

    def __init__(self, values, labels, problem_type: ProblemType):
        self.values = values
        self.labels = labels
        self.example_set = ExampleSetFactory(problem_type).make(values, labels)

    def get_data(self) -> tuple:
        return self.values, self.labels


def load_arff_to_example_set(
    path: str, label_attribute: str, problem_type: ProblemType
) -> ExampleSetWrapper:
    with open(path, "r") as file:
        content = file.read().replace('"', "'")
        arff_file = io.StringIO(content)
    arff_data_frame = read_arff(arff_file)

    attributes_names = []
    for column_name in arff_data_frame.columns:
        if column_name != label_attribute:
            attributes_names.append(column_name)

    values = arff_data_frame[attributes_names]
    labels = arff_data_frame[label_attribute]

    for column in values:
        _fix_missing_values(values[column])
    _fix_missing_values(labels)
    return ExampleSetWrapper(values, labels, problem_type)


def get_dataset_path(name: str) -> str:
    return f"{DATASETS_PATH}/{name}"


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

    def __init__(self, problem_type: ProblemType):
        self.param_config: dict[str, object] = None
        self._reference_report: TestReport = None
        self._example_set: ExampleSetWrapper = None
        self.induction_params: dict = None
        self.knowledge: Knowledge = None
        self.problem_type: ProblemType = problem_type

        self.name: str = None
        self.data_set_file_path: str = None
        self.label_attribute: str = None
        self.survival_time: str = None
        self.report_file_path: str = None
        self.using_existing_report_file: bool = False

    @property
    def example_set(self) -> ExampleSetWrapper:
        if self._example_set is None:
            self._example_set = load_arff_to_example_set(
                self.data_set_file_path, self.label_attribute, self.problem_type
            )
        return self._example_set

    @property
    def reference_report(self) -> TestReport:
        if self._reference_report is None:
            reader = TestReportReader(f"{self.report_file_path}.txt")
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
        self.parameter_configs: dict[str, dict[str, object]] = None
        self.datasets: list[DataSetConfig] = None


class TestConfigParser:
    TEST_KEY = "test"
    NAME_KEY = "name"
    IN_FILE_KEY = "in_file"
    TRAINING_KEY = "training"
    TRAIN_KEY = "train"
    LABEL_KEY = "label"
    DATASET_KEY = "dataset"
    DATASETS_KEY = "datasets"
    PARAM_KEY = "param"
    PARAMETERS_SET_KEY = "parameter_sets"
    PARAMETERS_KEY = "parameter_set"
    ENTRY_KEY = "entry"
    SURVIVAL_TIME_ROLE = "survival_time"

    EXPERTS_RULES_PARAMETERS_NAMES = [
        "expert_rules",
        "expert_preferred_conditions",
        "expert_forbidden_conditions",
    ]

    def __init__(self):
        self.tests_configs: dict[str, TestConfig] = {}
        self.root: ElementTree = None

    def _parse_survival_time(self, element) -> Union[str, None]:
        survival_time_element = element.find(TestConfigParser.SURVIVAL_TIME_ROLE)
        if element.find(TestConfigParser.SURVIVAL_TIME_ROLE) is not None:
            return survival_time_element.text
        else:
            return None

    def _parse_experts_rules_parameters(self, elements) -> list[tuple]:
        expert_rules = []
        for element in elements:
            rule_name: str = element.attrib["name"]
            rule_content: str = element.text
            # RuleKit originally used XML for specifying parameters and uses special xml characters
            rule_content = rule_content.replace("&lt;", "<").replace("&gt;", ">")
            expert_rules.append((rule_name, rule_content))
        return expert_rules if len(expert_rules) > 0 else None

    def _check_ambigous_data_sets_names(self, data_sets_configs: list[DataSetConfig]):
        dictionary = {}
        for element in data_sets_configs:
            dictionary[element.name] = None
        if len(dictionary.keys()) < len(data_sets_configs):
            raise ValueError("Datasets are ambigous")

    def _parse_data_set(self, element) -> DataSetConfig:
        data_set_config = DataSetConfig()
        data_set_config.label_attribute = element.find(TestConfigParser.LABEL_KEY).text
        train_element = element.find(TestConfigParser.TRAINING_KEY)
        train_element = train_element.find(TestConfigParser.TRAIN_KEY)
        data_set_config.train_file_name = train_element.find(
            TestConfigParser.IN_FILE_KEY
        ).text
        data_set_config.name = element.attrib.get("name", None)
        if data_set_config.name is None:
            file_name = os.path.basename(data_set_config.train_file_name)
            data_set_config.name = file_name.split(".")[0]
        data_set_config.survival_time = self._parse_survival_time(element)
        return data_set_config

    def _parse_data_sets(self, element) -> list[DataSetConfig]:
        data_set_configs = []
        node = element.find(TestConfigParser.DATASETS_KEY)
        for element in node.findall(TestConfigParser.DATASET_KEY):
            data_set_configs.append(self._parse_data_set(element))
        return data_set_configs

    def parse_test_parameters(self, element) -> dict[str, object]:
        params = {}
        for param_node in element.findall(TestConfigParser.PARAM_KEY):
            name: str = param_node.attrib["name"]
            if name in TestConfigParser.EXPERTS_RULES_PARAMETERS_NAMES:
                value = self._parse_experts_rules_parameters(
                    param_node.findall(TestConfigParser.ENTRY_KEY)
                )
            else:
                value = param_node.text
            params[name] = value
        return params

    def _parse_test_parameters_sets(self, element) -> dict[str, dict[str, object]]:
        parameters_sets = {}
        params_sets_node = element.findall(TestConfigParser.PARAMETERS_SET_KEY)[0]
        for param_set in params_sets_node.findall(TestConfigParser.PARAMETERS_KEY):
            name: str = param_set.attrib["name"]
            parameters_sets[name] = self.parse_test_parameters(param_set)
        return parameters_sets

    def _parse_test(self, element) -> TestConfig:
        test_config = TestConfig()
        test_config.parameter_configs = self._parse_test_parameters_sets(element)
        test_config.datasets = self._parse_data_sets(element)
        test_config.name = element.attrib["name"]
        return test_config

    def parse(self, file_path: str) -> dict[str, TestConfig]:
        self.tests_configs = {}
        self.root = ElementTree.parse(file_path).getroot()
        if self.root.tag == "test":
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
        params: dict[str, object],
        data_set_config: DataSetConfig,
        problem_type: ProblemType,
    ) -> TestCase:
        test_case = TestCase(problem_type)
        self._fix_params_typing(params)
        self._fix_deprecated_params(params)
        test_case.induction_params = params
        test_case.data_set_file_path = (
            f"{DATA_IN_DIRECTORY_PATH}/" f"{data_set_config.train_file_name}"
        )
        test_case.label_attribute = data_set_config.label_attribute
        test_case.name = test_case_name
        test_case.param_config = params
        return test_case

    def _fix_deprecated_params(self, params: dict[str, object]):
        deprecated_minsupp_new_name = "min_rule_covered"
        if deprecated_minsupp_new_name in params:
            params["minsupp_new"] = params.pop(deprecated_minsupp_new_name)

    def _fix_params_typing(self, params: dict):
        for key, value in params.items():
            if value == "false":
                params[key] = False
                continue
            if value == "true":
                params[key] = True
                continue
            if not "measure" in key:
                params[key] = int(float(value))

    def make(
        self,
        tests_configs: dict[str, TestConfig],
        report_dir_path: str,
        problem_type: ProblemType,
    ) -> list[TestCase]:
        test_cases = []
        for key in tests_configs.keys():
            test_config = tests_configs[key]
            for config_name in test_config.parameter_configs.keys():
                for data_set_config in test_config.datasets:
                    params = test_config.parameter_configs[config_name]
                    test_case_name = f"{key}.{config_name}.{data_set_config.name}"
                    test_config.parameter_configs[config_name].pop("use_expert", None)
                    expert_rules = test_config.parameter_configs[config_name].pop(
                        "expert_rules", None
                    )
                    preferred_conditions = test_config.parameter_configs[
                        config_name
                    ].pop("expert_preferred_conditions", None)
                    forbidden_conditions = test_config.parameter_configs[
                        config_name
                    ].pop("expert_forbidden_conditions", None)
                    test_case = self._make_test_case(
                        test_case_name,
                        test_config.parameter_configs[config_name],
                        data_set_config,
                        problem_type,
                    )
                    if "use_report" in params:
                        report_file_name = params["use_report"]
                        test_case.using_existing_report_file = True
                    else:
                        report_file_name = test_case_name
                    report_path = f"{report_dir_path}/{report_file_name}"
                    test_case.report_file_path = report_path
                    test_case.survival_time = data_set_config.survival_time
                    if (
                        expert_rules is not None
                        or preferred_conditions is not None
                        or forbidden_conditions is not None
                    ):
                        test_case.knowledge = Knowledge()
                        if expert_rules is not None:
                            test_case.knowledge.expert_rules = expert_rules
                        if forbidden_conditions is not None:
                            test_case.knowledge.expert_forbidden_conditions = (
                                forbidden_conditions
                            )
                        if preferred_conditions is not None:
                            test_case.knowledge.expert_preferred_conditions = (
                                preferred_conditions
                            )
                    test_cases.append(test_case)
        return test_cases


def get_rule_string(rule) -> str:
    return re.sub(r"(\\[[^\\]]*\\]$)|(\\([^\\)]*\\)$)", "", str(rule))


class TestReportReader:

    def __init__(self, file_name: str):
        self.file_name = file_name
        self._file = open(file_name, encoding="utf-8", mode="r")

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
            line = re.sub(r"\t", "", line)
            line = line.replace("\n", "")
            if line == REPORTS_SECTIONS_HEADERS["RULES"]:
                self._read_rules(test_report)
            elif line == "":
                continue
            else:
                raise ValueError(
                    f"Invalid report file format for file: {self.file_name}"
                )
        return test_report

    def close(self):
        self._file.close()


class TestReportWriter:

    def __init__(self, file_name: str):
        self._file = open(file_name, encoding="utf-8", mode="w")
        if not os.path.exists(REPORTS_OUT_DIRECTORY_PATH):
            os.makedirs(REPORTS_OUT_DIRECTORY_PATH)

    def write(self, rule_set):
        self._file.write("\n")
        self._file.write(f'{REPORTS_SECTIONS_HEADERS["RULES"]}\n')
        for rule in rule_set.rules:
            self._file.write(f"\t{get_rule_string(rule)}")

    def close(self):
        self._file.close()


def get_test_cases(class_name: str) -> list[TestCase]:
    if not os.path.exists(DATA_IN_DIRECTORY_PATH):
        raise RuntimeError(
            """\n
Test resources directory dosen't exist. Check if 'tests/resources/' directory exist.

If you're running tests for the first time you need to download resources folder from RuleKit repository by running:
    python tests/resources.py download
        """
        )
    problem_type: ProblemType = _get_problem_type_from_test_case_class_name(class_name)
    configs = TestConfigParser().parse(f"{TEST_CONFIG_PATH}/{class_name}.xml")
    return TestCaseFactory().make(
        configs, f"{REPORTS_IN_DIRECTORY_PATH}/{class_name}/", problem_type
    )


def _get_problem_type_from_test_case_class_name(class_name: str) -> ProblemType:
    class_name = class_name.lower()
    if "regression" in class_name:
        return ProblemType.REGRESSION
    elif "survival" in class_name:
        return ProblemType.SURVIVAL
    elif "classification" in class_name:
        return ProblemType.CLASSIFICATION
    raise Exception(f"Unknown problem type for test case class name: {class_name}")


def assert_rules_are_equals(expected: list[str], actual: list[str]):
    def sanitize_rule_string(rule_string: str) -> str:
        return re.sub(r"(\t)|(\n)|(\[[^\]]*\]$)", "", rule_string)

    expected = list(map(sanitize_rule_string, expected))
    actual = list(map(sanitize_rule_string, actual))

    if len(expected) != len(actual):
        raise AssertionError(
            "Rulesets have different number of rules, actual: "
            f'{len(actual)}, expected: {len(expected)}'
        )
    dictionary = {}
    for rule in expected:
        dictionary[rule] = 0
    for rule in actual:
        key = rule
        if key in dictionary:
            dictionary[key] = dictionary[key] + 1
        else:
            raise AssertionError(
                "Actual ruleset contains rules not present in expected ruleset"
            )
    for value in dictionary.values():
        if value == 0:
            raise AssertionError("Ruleset are not equal, some rules are missing")
        elif value > 1:
            raise AssertionError("Somes rules were duplicated")


def assert_accuracy_is_greater(prediction, expected, threshold: float):
    labels = expected.to_numpy().astype(str)
    acc = metrics.accuracy_score(labels, prediction)
    if acc <= threshold:
        raise AssertionError(f"Accuracy should be greater than {threshold} (was {acc})")


def assert_score_is_greater(prediction, expected, threshold: float):
    if isinstance(prediction[0], int):
        labels = expected.to_numpy().astype(int)
    elif isinstance(prediction[0], float):
        labels = expected.to_numpy().astype(float)
    else:
        raise ValueError(
            f"Invalid prediction type: {str(type(prediction[0]))}. "
            + "Supported types are: 1 dimensional numpy array or pandas Series object."
        )
    explained_variance_score = metrics.explained_variance_score(labels, prediction)

    if explained_variance_score <= threshold:
        raise AssertionError(f"Score should be greater than {threshold}")
