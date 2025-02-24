from exception import VmatchException
from functools import wraps

_model_name = "log"


class LogException(VmatchException):
    """milvus exception"""

    pass


def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LogException as e:
            raise e
        except Exception as e:
            raise LogException(f"{_model_name} have exception: {e}")

    return wrapper
