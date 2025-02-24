from functools import wraps

from exception import VmatchException


class ProcessException(VmatchException):
    """process exception"""

    pass


def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            raise ProcessException(f"os error: {e}") from e
        except ValueError as e:
            raise ProcessException(f"value error: {e}") from e
        except RuntimeError as e:
            raise ProcessException(f"runtime error: {e}") from e
        except ProcessException as e:
            raise e from e
        except Exception as e:
            raise ProcessException(f"error: {e}") from e

    return wrapper
