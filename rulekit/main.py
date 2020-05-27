import os
import jpype
import jpype.imports

from typing import List
import glob
import logging
import zipfile
import re


class RuleKit:

    version: str

    logger = None
    _jar_dir_path: str
    _class_path: str
    _rulekit_jar_file_path: str

    @staticmethod
    def init():
        RuleKit._setup_logger()
        current_path: str = os.path.dirname(os.path.realpath(__file__))
        RuleKit._jar_dir_path = f"{current_path}/jar"
        try:
            jars_paths: List[str] = glob.glob(f"{RuleKit._jar_dir_path}\\*.jar")
            RuleKit._class_path = f'{str.join(";", jars_paths)}'
            RuleKit._rulekit_jar_file_path = list(filter(lambda path: 'rulekit' in os.path.basename(path), jars_paths))[0]
            print(RuleKit._rulekit_jar_file_path)
        except IndexError as error:
            RuleKit.logger.error('Failed to load jar files')
            raise error
        RuleKit._read_versions()
        RuleKit._launch_jvm()

    @staticmethod
    def _setup_logger():
        logging.basicConfig()
        RuleKit.logger = logging.getLogger('RuleKit')

    @staticmethod
    def _read_versions():
        jar_archive = zipfile.ZipFile(RuleKit._rulekit_jar_file_path, 'r')
        try:
            manifest_file_content: str = jar_archive.read('META-INF/MANIFEST.MF').decode('utf-8')
            RuleKit.version = re.findall(r'Implementation-Version: \S+\r', manifest_file_content)[0].split(' ')[1]
        except Exception as error:
            RuleKit.logger.error('Failed to read RuleKit versions from jar file')
            RuleKit.logger.error(error)
            raise error

    @staticmethod
    def _launch_jvm():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=%s" % f'{RuleKit._class_path}', convertStrings=False)


if __name__ == "__main__":
    # Launch the JVM
    RuleKit.init()

    from tree.decision_tree_classifier import DecisionTreeClassifier

    clf = DecisionTreeClassifier()

    X = [['0', '1'], ['1', '0'], ['1', '1'], ['0', '0']]
    Y = ['0', '0', '1', '0']
    clf = clf.fit(X, Y)
    res = clf.predict([['1', '1'], ['1', '1'], ['0', '0']])
    print(res)
