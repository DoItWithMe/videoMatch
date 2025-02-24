from exception import VmatchException
from functools import wraps

_model_name = "ffmpeg"


class FFmpegException(VmatchException):
    """ffmpeg exception"""

    pass


class FFmpegEofException(VmatchException):
    """ffmpeg exception"""

    pass


def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (OSError, ValueError, FFmpegException) as e:
            raise FFmpegException(f"{_model_name} get exception: {e}")
        except FFmpegEofException as e:
            raise e
        except Exception as e:
            raise FFmpegException(f"{_model_name} get an unexpect exception: {e}")

    return wrapper


def exception_ignore(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            pass

    return wrapper
