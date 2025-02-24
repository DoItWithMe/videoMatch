import threading

def thread_safe_singleton(cls):
    instances = {}
    lock = threading.Lock()  # 创建一个锁对象

    def get_instance(*args, **kwargs):
        with lock:  # 确保线程安全
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance