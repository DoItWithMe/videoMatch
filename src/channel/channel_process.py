from .exception import ChannelException, exception_handler
from multiprocessing.synchronize import Event as EventCls
from multiprocessing import Queue, Event
from typing import Any


class Channel:
    @exception_handler
    def __init__(self, max_size=0) -> None:
        self.queue: Queue = Queue(max_size)
        self.closed: EventCls = Event()

    @exception_handler
    def send(self, item) -> None:
        if self.is_closed():
            raise ChannelException("Cannot send to a closed channel")
        self.queue.put(item)

    @exception_handler
    def recv(self, timeout=None) -> Any:
        if self.is_closed() and self.queue.empty():
            raise ChannelException("Channel is closed and no more items are available")

        return self.queue.get(timeout=timeout)

    @exception_handler
    def close(self) -> None:
        self.closed.set()

    @exception_handler
    def is_closed(self) -> bool:
        return self.closed.is_set()
