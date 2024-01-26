"""Contains classes for initializing RuleKit Java backend
"""
import os
import glob
import logging
import zipfile
import re
from enum import Enum
from subprocess import Popen, PIPE, STDOUT
import jpype
import jpype.imports


class JRE_Type(Enum):  # pylint: disable=invalid-name
    """:meta private:"""
    OPEN_JDK = 'open_jdk'
    ORACLE = 'oracle'


class RuleKit:
    """Class used for initializing RuleKit. It starts JVM underhood and setups it with jars. 

    .. note:: Since version 1.7.0 there is no need to manually initialize RuleKit. 
    You may just skip the **RuleKit.init()** line. However in certain scenarios when
    you want use a custom RuleKit jar file or modify Java VM parameters, this class
    can be used.


    Attributes
    ----------
    version : str
        version of RuleKit jar used by wrapper (not equal to python package version).
    """
    version: str
    _logger = None
    _jar_dir_path: str
    _class_path: str
    _rulekit_jar_file_path: str
    _jre_type: JRE_Type
    initialized: bool = False

    @staticmethod
    def _detect_jre_type():
        try:
            output = Popen(["java", "-version"], stderr=STDOUT, stdout=PIPE)
            output = str(output.communicate()[0])
            if 'openjdk' in output:
                RuleKit._jre_type = JRE_Type.OPEN_JDK
            else:
                RuleKit._jre_type = JRE_Type.ORACLE
        except FileNotFoundError as error:
            raise RuntimeError(
                'RuletKit requires java JRE to be installed (version 1.8.0 recommended)'
            ) from error

    @staticmethod
    def init(
        initial_heap_size: int = None,
        max_heap_size: int = None,
        rulekit_jar_file_path: str = None
    ):
        """Initialize package.

        This method configure and starts JVM and load RuleKit jar file. 

        .. note:: Since version 1.7.0 it don't have to be called before using any operator class. 
        However in certain scenarios when you want use a custom RuleKit jar file or modify Java VM
        parameters, this method can be used.

        Parameters
        ----------
        initial_heap_size : int
            JVM initial heap size in mb
        max_heap_size : int
            JVM max heap size in mb
        rulekit_jar_file_path : str
            Path to the RuleKit jar file. This parameters.
            .. note::
                You probably don't need to use this parameter unless you want to use 
                your own custom version of RuleKit jar file. Otherwise leave it as it
                is and the package will use the official RuleKit release jar file.

        Raises
        ------
        Exception
            If failed to load RuleKit jar file.
        """
        if RuleKit.initialized:
            return
        RuleKit._setup_logger()

        RuleKit._detect_jre_type()
        current_path: str = os.path.dirname(os.path.realpath(__file__))
        RuleKit._jar_dir_path = f"{current_path}/jar"
        class_path_separator = os.pathsep
        try:
            jars_paths: list[str] = glob.glob(f"{RuleKit._jar_dir_path}/*.jar")
            RuleKit._rulekit_jar_file_path = list(
                filter(lambda path: 'rulekit' in os.path.basename(path), jars_paths)
            )[0]
            if rulekit_jar_file_path is not None:
                jars_paths.remove(RuleKit._rulekit_jar_file_path)
                jars_paths.append(rulekit_jar_file_path)
                RuleKit._rulekit_jar_file_path = rulekit_jar_file_path
            RuleKit._class_path = f'{str.join(class_path_separator, jars_paths)}'
        except IndexError as error:
            RuleKit._logger.error('Failed to load jar files')
            raise RuntimeError(f'''\n
Failed to load RuleKit jar file. Check if valid rulekit jar file is present in "{RuleKit._jar_dir_path}" directory.

If you're running this package for the first time you need to download RuleKit jar file by running:
    python -m rulekit download_jar
        ''') from error
        RuleKit._read_versions()
        RuleKit._launch_jvm(initial_heap_size, max_heap_size)
        RuleKit.initialized = True

    @staticmethod
    def _setup_logger():
        logging.basicConfig()
        RuleKit._logger = logging.getLogger('RuleKit')

    @staticmethod
    def _read_versions():
        jar_archive = zipfile.ZipFile(RuleKit._rulekit_jar_file_path, 'r')
        try:
            manifest_file_content: str = jar_archive.read(
                'META-INF/MANIFEST.MF').decode('utf-8')
            RuleKit.version = re.findall(
                r'Implementation-Version: \S+\r', manifest_file_content)[0].split(' ')[1]
        except Exception as error:
            RuleKit._logger.error(
                'Failed to read RuleKit versions from jar file')
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
            jpype.startJVM(jpype.getDefaultJVMPath(), *
                           params, convertStrings=False)
