from datetime import datetime


class TimeRecorder:
    def __init__(self) -> None:
        self._start_time_list = list()
        self._end_time_list = list()

    def start_record(self):
        self._start_time_list.append(datetime.now())

    def end_record(self):
        self._end_time_list.append(datetime.now())

    def get_total_duration_miliseconds(self):
        ms_duration = 0
        for i in range(min(len(self._start_time_list), len(self._end_time_list))):
            ms_duration += int(
                (self._end_time_list[i] - self._start_time_list[i]).total_seconds()
                * 1000
            )

        return ms_duration

    def get_avg_duration_miliseconds(self):
        count = min(len(self._start_time_list), len(self._end_time_list))
        if count == 0:
            return 0

        return self.get_total_duration_miliseconds() // count

    def clear_records(self):
        self._start_time_list.clear()
        self._end_time_list.clear()

def milliseconds_to_hhmmss(ms: int):
    # calculate total seconds
    total_seconds = ms / 1000.0

    # calculate hour, minute, seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # format
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


def hhmmss_to_milliseconds(hhmmss: str) -> int:
    # Split the input string by ':'
    hours, minutes, seconds = map(int, hhmmss.split(":"))

    # Convert everything to milliseconds
    total_milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000

    return total_milliseconds