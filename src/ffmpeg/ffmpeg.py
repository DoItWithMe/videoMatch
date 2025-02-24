import subprocess
import os
import shutil
import signal
from .exception import (
    exception_handler,
    exception_ignore,
    FFmpegException,
    FFmpegEofException,
)
from typing import Generator, Optional, Any
import threading
from io import BufferedReader
import queue
import io


class FFmpeg:
    __JPEG_START_MARKER = b"\xff\xd8"
    __JPEG_END_MARKER = b"\xff\xd9"
    __READ_BUFFER_SIZE = 4 * 1024

    @exception_handler
    def __init__(
        self,
        ffmpeg_bin_path: str,
        input_file_path: str,
        imgs_save_dir: Optional[str],
        fps: int,
    ) -> None:
        if not os.path.exists(ffmpeg_bin_path):
            raise FFmpegException(f"{ffmpeg_bin_path} not exists")

        if not os.path.exists(input_file_path):
            raise FFmpegException(f"{input_file_path} not exists")

        self.__ffmpeg_bin_path: str = ffmpeg_bin_path
        self.__input_file_path: str = input_file_path
        self.__imgs_save_dir: Optional[str] = imgs_save_dir

        if self.__imgs_save_dir is not None:
            self.__imgs_save_dir = os.path.join(
                self.__imgs_save_dir,
                f"{os.path.splitext(os.path.basename(input_file_path))[0]}",
            )
            shutil.rmtree(self.__imgs_save_dir, ignore_errors=True)
            os.makedirs(self.__imgs_save_dir, exist_ok=True)

        self.__command: str = (
            f"{self.__ffmpeg_bin_path} -i {self.__input_file_path} "
            f"-loglevel error -nostdin -y -vf fps={fps} -start_number 0 "
            f"-q 0 -f image2pipe "
            f"-vcodec mjpeg - "
        )

        self.__ffmepg_p: Optional[subprocess.Popen[bytes]] = None

        self.__read_thread: Optional[threading.Thread] = None
        self.__cache_queue: queue.Queue[io.BytesIO] = queue.Queue()

        self.__eof: bool = False

    @exception_ignore
    def __inner_read_worker(
        self,
        read_pipe: Optional[BufferedReader],
    ) -> None:
        if read_pipe is None:
            raise FFmpegException("read pipe is none")

        buffer = bytearray()
        in_image = False

        while True:

            chunk = read_pipe.read(self.__READ_BUFFER_SIZE)
            if not chunk:
                self.__eof = True
                break
            buffer.extend(chunk)
            while True:
                if not in_image:
                    start_index = buffer.find(self.__JPEG_START_MARKER)
                    if start_index != -1:
                        in_image = True
                        buffer = buffer[start_index:]
                    else:
                        buffer.clear()
                        break
                else:
                    end_index = buffer.find(self.__JPEG_END_MARKER)
                    if end_index != -1:
                        jpeg_data: bytearray = buffer[: end_index + 2]
                        self.__cache_queue.put(io.BytesIO(jpeg_data))

                        buffer = buffer[end_index + 2 :]
                        in_image = False
                    else:
                        break

    @exception_handler
    def start(self) -> None:
        if self.__ffmepg_p is not None or self.__read_thread is not None:
            raise FFmpegException("repeated starting is not allowed")

        self.__ffmepg_p = subprocess.Popen(
            self.__command,
            bufsize=self.__READ_BUFFER_SIZE,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            preexec_fn=os.setsid,
        )

        self.__read_thread = threading.Thread(
            target=self.__inner_read_worker,
            name="jpeg buffer reader",
            args=(self.__ffmepg_p.stdout,),
        )

        self.__read_thread.daemon = True

        self.__read_thread.start()

    @exception_handler
    def stop(self) -> None:
        if self.__ffmepg_p is None or self.__read_thread is None:
            raise FFmpegException("ffmpeg not start")

        os.killpg(os.getpgid(self.__ffmepg_p.pid), signal.SIGTERM)
        self.__read_thread.join()

    @exception_handler
    def get_img(self) -> Optional[io.BytesIO]:
        if self.__read_thread is None or not self.__read_thread.is_alive():
            if not self.__eof:
                raise FFmpegException("ffmpeg not start")

        if self.__cache_queue.empty() and self.__eof:
            return None

        return self.__cache_queue.get()
