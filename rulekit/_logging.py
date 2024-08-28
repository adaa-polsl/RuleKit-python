class _RuleKitJavaLoggerConfig:
    """Class storing configuration of the RuleKit java logger.

    Raises:
        ValueError: for invalid configuration
    """
    _MIN_VERBOSITY_LEVEL: int = 1
    _MAX_VERBOSITY_LEVEL: int = 2

    def __init__(self, log_file_path: str, verbosity_level: int):
        self.log_file_path: int = log_file_path
        self.verbosity_level: str = self._map_verbosity_level_to_value(
            verbosity_level
        )

    def _map_verbosity_level_to_value(self, verbosity_level: int) -> str:
        if (
            verbosity_level < self._MIN_VERBOSITY_LEVEL or
            verbosity_level > self._MAX_VERBOSITY_LEVEL
        ):
            raise ValueError(
                'Invalid verbosity level. Allowed values are from ' +
                f'{self._MIN_VERBOSITY_LEVEL} to {self._MAX_VERBOSITY_LEVEL}'
            )
        return 'v' * verbosity_level
