from jpype import JException


class RuleKitJavaException(Exception):

    def __init__(self, java_exception: JException) -> None:
        self.original_exception: JException = java_exception
        self.message: str = java_exception.message()
        super().__init__(self.message)

    def print_java_stack_trace(self) -> None:
        print(self.original_exception.stacktrace())
