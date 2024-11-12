"""Module containing classes for handling exceptions."""
from typing import Any
from typing import Callable

from jpype import JException


class RuleKitJavaException(Exception):
    """Wrapper method for handling Java exceptions produced by RuleKit.
    Useful for debugging internal Java code error. It allows to access java
    stack traces easily.

    Example:
        >>> from rulekit import RuleKit
        >>> from rulekit.classification import RuleClassifier
        >>>
        >>> clf = RuleClassifier()
        >>> try:
        >>>     clf.fit(X, y)
        >>> except JException as e: # catch java exceptions
        >>>     e = RuleKitJavaException(e) # wrap them for easier handling
        >>>     e.print_java_stack_trace() # access java stack trace
    """

    def __init__(self, java_exception: JException) -> None:
        """
        Args:
            java_exception (JException): original java exception
        """
        self._original_exception: JException = java_exception
        self.message: str = java_exception.message()
        super().__init__(self.message)

    @property
    def original_exception(self) -> JException:
        """
        Original java exception
        Returns:
            JException: original java exception
        """
        return self._original_exception

    def print_java_stack_trace(self) -> None:
        """Prints java exception stack trace."""
        print(self.original_exception.stacktrace())


class RuleKitMisconfigurationException(Exception):
    """Exception indicating that some RuleKit parameters are misconfigured -
    their values were not correctly passed to java code.
    """

    def __init__(
        self,
        java_parameters: dict[str, Any],
        python_parameters: dict[str, Any]
    ) -> None:

        super().__init__(
            self._prepare_message(java_parameters, python_parameters)
        )
        self._java_parameters: dict[str, Any] = java_parameters
        self._python_parameters: dict[str, Any] = python_parameters

    def _prepare_message(
        self,
        java_parameters: dict[str, Any],
        python_parameters: dict[str, Any]
    ) -> str:
        combined_keys: set[str] = set(
            java_parameters.keys()) | set(python_parameters.keys())
        params_lines: list[str] = []
        for key in combined_keys:
            java_value = java_parameters.get(key)
            python_value = python_parameters.get(key)
            line: str = f'  {key}: ({java_value},  {python_value}),'
            # skip check for user defined measures
            skip_check: bool = isinstance(python_value, Callable)
            if java_value != python_value and not skip_check:
                line = f'{line} <-- **DIFFERENT**'
            params_lines.append(line)
        message: str = (
            'RuleKit parameters configuration error' +
            'RuleGenerator parameters configured in Java do not' +
            'match with given parameters\n\n' +
            'Parameters (first value is in Java, second is in Python):\n{\n' +
            '\n'.join(params_lines) +
            '\n}'
        )
        return message

    @property
    def java_parameters(self) -> dict[str, Any]:
        """
        Configured operator parameters values extracted from java
        Returns:
            dict[str, Any]: parameters
        """
        return self._java_parameters

    @property
    def python_parameters(self) -> dict[str, Any]:
        """
        Operator parameters passed to the python operator class
        Returns:
            dict[str, Any]: parameters
        """
        return self._python_parameters
