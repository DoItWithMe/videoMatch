from exception import VmatchException
from functools import wraps

_model_name = "configs"


class ConfigsException(VmatchException):
    """configs exception"""

    pass


def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            raise ConfigsException(f"{_model_name} have os exception: {e}")
        except ValueError as e:
            raise ConfigsException(f"{_model_name} have input value error: {e}")
        except ConfigsException as e:
            raise e
        except Exception as e:
            raise ConfigsException(f"{_model_name} have exception: {e}")

    return wrapper
