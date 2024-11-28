"""Contains classes for initializing RuleKit Java backend
"""
import logging
import os
import re
import zipfile
from enum import Enum
from subprocess import PIPE
from subprocess import Popen
from subprocess import STDOUT
from typing import Optional

import jpype.imports

from rulekit._logging import _RuleKitJavaLoggerConfig

__RULEKIT_RELEASE_VERSION__ = "2.1.24"
__VERSION__ = f"{__RULEKIT_RELEASE_VERSION__}.0"


class JRE_Type(Enum):  # pylint: disable=invalid-name
    """:meta private:"""

    OPEN_JDK = "open_jdk"
    ORACLE = "oracle"


class RuleKit:
    """Class used for initializing RuleKit. It starts JVM underhood and setups it
    with jars.

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
    _logger: logging.Logger = None
    _jar_dir_path: str
    _class_path: str
    _rulekit_jar_file_path: str
    _jre_type: JRE_Type
    _java_logger_config: Optional[_RuleKitJavaLoggerConfig] = None
    initialized: bool = False

    @staticmethod
    def _detect_jre_type():
        try:
            with Popen(["java", "-version"], stderr=STDOUT, stdout=PIPE) as output:
                output = str(output.communicate()[0])
                if "openjdk" in output:
                    RuleKit._jre_type = JRE_Type.OPEN_JDK
                else:
                    RuleKit._jre_type = JRE_Type.ORACLE
        except FileNotFoundError as error:
            raise RuntimeError(
                "RuletKit requires java JRE to be installed (version 1.8.0 recommended)"
            ) from error

    @staticmethod
    def init(
        initial_heap_size: int = None,
        max_heap_size: int = None,
        rulekit_jar_file_path: str = None,
    ):
        """Initialize package.

        This method configure and starts JVM and load RuleKit jar file.

        .. note:: Since version 1.7.0 it don't have to be called before using any
        operator class. However in certain scenarios when you want use a custom RuleKit
        jar file or modify Java VM parameters, this method can be used.

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
            jar_file_name: str = "rulekit-" + f"{__RULEKIT_RELEASE_VERSION__}-all.jar"
            RuleKit._rulekit_jar_file_path = os.path.join(
                RuleKit._jar_dir_path, jar_file_name
            )
            jars_paths: list[str] = [RuleKit._rulekit_jar_file_path]
            if rulekit_jar_file_path is not None:
                jars_paths.remove(RuleKit._rulekit_jar_file_path)
                jars_paths.append(rulekit_jar_file_path)
                RuleKit._rulekit_jar_file_path = rulekit_jar_file_path
            RuleKit._class_path = f"{str.join(class_path_separator, jars_paths)}"
        except IndexError as error:
            RuleKit._logger.error("Failed to load jar files")
            raise RuntimeError(
                f"""\n
Failed to load RuleKit jar file. Check if valid rulekit jar file is present in
"{RuleKit._jar_dir_path}" directory.

If you're running this package for the first time you need to download RuleKit jar
file by running:
    python -m rulekit download_jar
        """
            ) from error
        RuleKit._read_versions()
        RuleKit._launch_jvm(initial_heap_size, max_heap_size)
        RuleKit.initialized = True

    @staticmethod
    def _setup_logger():
        logging.basicConfig()
        RuleKit._logger = logging.getLogger("RuleKit")

    @staticmethod
    def _read_versions():
        with zipfile.ZipFile(RuleKit._rulekit_jar_file_path, "r") as jar_archive:
            try:
                manifest_file_content: str = jar_archive.read(
                    "META-INF/MANIFEST.MF"
                ).decode("utf-8")
                RuleKit.version = re.findall(
                    r"Implementation-Version: \S+\r", manifest_file_content
                )[0].split(" ")[1]
            except Exception as error:
                RuleKit._logger.error("Failed to read RuleKit versions from jar file")
                RuleKit._logger.error(error)
                raise error

    @staticmethod
    def _launch_jvm(initial_heap_size: int, max_heap_size: int):
        if jpype.isJVMStarted():
            RuleKit._logger.info("JVM already running")
        else:
            params = [
                f"-Djava.class.path={RuleKit._class_path}",
            ]
            if initial_heap_size is not None:
                params.append(f"-Xms{initial_heap_size}m")
            if max_heap_size is not None:
                params.append(f"-Xmx{max_heap_size}m")
            jpype.startJVM(jpype.getDefaultJVMPath(), *params, convertStrings=False)

    @staticmethod
    def configure_java_logger(
        log_file_path: str,
        verbosity_level: int = 1,
    ):
        """Enable Java debug logging. You probably don't need to use this
        method unless you want too deep dive into the process of rules inductions
         or your're debugging some issues.


        Args:
            log_file_path (str): Path to the file where logs will be stored
            verbosity_level (int, optional): Verbosity level.
                Minimum value is 1, maximum value is 2, default value is 1.
        """
        RuleKit._java_logger_config = _RuleKitJavaLoggerConfig(
            verbosity_level=verbosity_level, log_file_path=log_file_path
        )

    @staticmethod
    def get_java_logger_config() -> Optional[_RuleKitJavaLoggerConfig]:
        """Returns the Java logger configuration configured using
        `configure_java_logger` method

        Returns:
            Optional[_RuleKitJavaLoggerConfig]: Java logger configuration
        """
        return RuleKit._java_logger_config
