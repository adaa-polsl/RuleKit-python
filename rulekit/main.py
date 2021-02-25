import os
import sys
import jpype
import jpype.imports

from typing import List
import glob
import logging
import zipfile
import re


class RuleKit:

    version: str
    _logger = None
    _jar_dir_path: str
    _class_path: str
    _rulekit_jar_file_path: str

    @staticmethod
    def init(initial_heap_size: int = None, max_heap_size: int = None):
        RuleKit._setup_logger()
        current_path: str = os.path.dirname(os.path.realpath(__file__))
        RuleKit._jar_dir_path = f"{current_path}/jar"
        try:
            jars_paths: List[str] = glob.glob(f"{RuleKit._jar_dir_path}/*.jar")
            RuleKit._class_path = f'{str.join(";", jars_paths)}'
            RuleKit._rulekit_jar_file_path = list(filter(lambda path: 'rulekit' in os.path.basename(path), jars_paths))[0]
        except IndexError as error:
            RuleKit._logger.error('Failed to load jar files')
            raise error
        RuleKit._read_versions()
        RuleKit._launch_jvm(initial_heap_size, max_heap_size)

    @staticmethod
    def _setup_logger():
        logging.basicConfig()
        RuleKit._logger = logging.getLogger('RuleKit')

    @staticmethod
    def _read_versions():
        jar_archive = zipfile.ZipFile(RuleKit._rulekit_jar_file_path, 'r')
        try:
            manifest_file_content: str = jar_archive.read('META-INF/MANIFEST.MF').decode('utf-8')
            RuleKit.version = re.findall(r'Implementation-Version: \S+\r', manifest_file_content)[0].split(' ')[1]
        except Exception as error:
            RuleKit._logger.error('Failed to read RuleKit versions from jar file')
            RuleKit._logger.error(error)
            raise error

    @staticmethod
    def _launch_jvm(initial_heap_size: int, max_heap_size: int):
        if jpype.isJVMStarted():
            RuleKit._logger.info('JVM already running')
        else:
            params = [
                f'-Djava.class.path={RuleKit._class_path}',
            ]
            if initial_heap_size is not None:
                params.append(f'-Xms{initial_heap_size}m')
            if max_heap_size is not None:
                params.append(f'-Xmx{max_heap_size}m')
            jpype.startJVM(jpype.getDefaultJVMPath(), *params, convertStrings=False)
