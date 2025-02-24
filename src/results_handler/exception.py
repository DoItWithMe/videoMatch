from exception import VmatchException
from functools import wraps

_model_name = "results handler"


class ResultsHandlerException(VmatchException):
    """milvus exception"""

    pass


def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ResultsHandlerException as e:
            raise e
        except Exception as e:
            raise ResultsHandlerException(f"{_model_name} have exception: {e}")

    return wrapper
