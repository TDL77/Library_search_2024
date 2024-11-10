import logging
import loguru
import sys


def add_loguru(logger: "loguru.Logger") -> None:
    """
    Remove a previously added handler and stop sending logs to its sink (default - stderr handler)
    """
    logger.remove()
    logger.add(sys.stdout, serialize=True)


class EndpointFilter(logging.Filter):
    """
    Filter class to exclude specific endpoints from log entries.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out log entries for excluded endpoints.

        :param record: The log record to be filtered.
        :return: bool: True if the log entry should be included, False otherwise.
        """
        if len(record.args) >= 3:
            return record.args[2] not in ["/healthcheck"]
        return False
    