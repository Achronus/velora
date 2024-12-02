class IncorrectFileTypeError(Exception):
    def __init__(self, message: str = "Incorrect file type provided.") -> None:
        super().__init__(message)
        self.message = message


class RunNotFoundError(Exception):
    def __init__(self, message: str = "Analytics run failed.") -> None:
        super().__init__(message)
        self.message = message


class RolloutsFullError(Exception):
    def __init__(self, message: str = "Rollouts full.") -> None:
        super().__init__(message)
        self.message = message
