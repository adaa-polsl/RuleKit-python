"""Module containing classes for handling exceptions."""
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
        self.original_exception: JException = java_exception
        self.message: str = java_exception.message()
        super().__init__(self.message)

    def print_java_stack_trace(self) -> None:
        """Print java exception stack trace."""
        print(self.original_exception.stacktrace())
