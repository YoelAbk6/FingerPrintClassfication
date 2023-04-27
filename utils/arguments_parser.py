import argparse
import logging

logger = logging.getLogger(__name__)


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        """Prints a usage message incorporating the message to logger and exits

        Arguments:
            message (str) -- error message

        """
        logger.error(message)
        self.print_help()
        exit(1)
