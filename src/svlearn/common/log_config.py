#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2023.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  #
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  #
#   Author: Asif Qamar
#  -------------------------------------------------------------------------------------------------
#

import logging
from rich.logging import RichHandler
import sys
from pathlib import Path
import pytest
from unittest.mock import patch

from svlearn.common.decorators import singleton

@singleton
class LogConfiguration:
    """
    This class is used to configure the logging for the application.

    Example:
        ```python
        log_config = LogConfiguration(with_console_logging=True,
                                        with_file_logging=True,
                                        log_file_path='logs/svlearn.log')
        log_config.set_level()

        logger = logging.getLogger('SUPPORTVECTORS-LLM-BOOTCAMP')
        ```
    
    Parameters:
        application_name: The name of the application.
        with_console_logging: A boolean value to enable console logging.
        with_file_logging: A boolean value to enable file logging.
        log_file_path: The path to the log file.

    Returns:
        None
    """
    # ---------------------------------------------------------------------------------------------
    __DEFAULT_APPLICATION_NAME: str = 'SUPPORTVECTORS-LLM-BOOTCAMP'
    __DEFAULT_LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # ---------------------------------------------------------------------------------------------
    
    def __init__(self,
                 application_name = __DEFAULT_APPLICATION_NAME, 
                 with_console_logging: bool = True,
                 with_file_logging: bool = False,
                 log_file_path: str = None,
                 log_format:str = __DEFAULT_LOG_FORMAT) -> None:
        #docstring
        """
        Initialize the LogConfiguration class.

        Example:
            ```python
            log_config = LogConfiguration(with_console_logging=True,
                                            with_file_logging=True,
                                            log_file_path='logs/svlearn.log')
            ```
        Parameters:
            application_name (str): The name of the application.
            with_console_logging (bool): A boolean value to enable console logging.
            with_file_logging (bool): A boolean value to enable file logging.
            log_file_path (str): The path to the log file.
            log_format (str): The format of the log message.
        """ 
        
        # Preconditions check
        if not application_name:
            raise ValueError('Must provide a non-empty value for "application_name" parameter.')

        if not log_format:
            raise ValueError('Must provide a non-empty and valid format value for "log_format" parameter.')

        if with_file_logging and not log_file_path:
            raise ValueError('Must provide a non-empty value for "log_file_path" '
            'parameter when "with_file_logging" is set to True.')
      
        if with_file_logging:
            path: Path = Path(log_file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)


        self.application_name = application_name
        self.with_console_logging = with_console_logging
        self.with_file_logging = with_file_logging
        self.log_file_path = log_file_path
        self.log_format = log_format
        self.setup_logging()

    # ---------------------------------------------------------------------------------------------
    def setup_logging(self, level: int =logging.INFO):
        """
        Set up the logging configuration.

        Example:
            ```python
            log_config = LogConfiguration(with_console_logging=True,
                                            with_file_logging=True,
                                            log_file_path='logs/svlearn.log')
            log_config.setup_logging(level=logging.INFO)
            ```
        
        Parameters:
        - level (int): The logging level you want to set for the application.
        """
         
        # Create a logger object
        logger = logging.getLogger(self.application_name)
        logger.setLevel(level)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(self.log_format)

        # Create Console log handler
        if self.with_console_logging:
            console_handler = RichHandler() #logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Create File log handler
        if self.with_file_logging:
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
	# Finally, we always want the RichHandler in place.
        logger.addHandler(RichHandler())


        # We want to log uncaught exceptions as well
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Call the default KeyboardInterrupt handler
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.error('\n' + '-'*80 +'\nUNCAUGHT EXCEPTION\n' + '-'*80, 
                            exc_info=(exc_type, exc_value, exc_traceback))
        sys.excepthook = handle_exception      

    # ---------------------------------------------------------------------------------------------

    def set_level(self, level: int = logging.INFO):
        """
        Set the logging level for the logger.

        Example:
            ```python
            log_config = LogConfiguration(with_console_logging=True,
                                            with_file_logging=True,
                                            log_file_path='logs/svlearn.log')
            log_config.set_level(level=logging.INFO)
            ```
        
        Parameters:
            level: The logging level you want to set for the application.

        Returns:
            None
        """
        logger = logging.getLogger(self.application_name)
        logger.setLevel(level)
        
# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    The main function of the application.
    """
    # Setup logging with the desired level
    log_config = LogConfiguration(with_console_logging=True, 
                                    with_file_logging=True,
                                    log_file_path='logs/svlearn.log')
    log_config.set_level()

    # Now you can use the logger within your main application
    logger = logging.getLogger('SUPPORTVECTORS-LLM-BOOTCAMP')
    logger.info("Application is starting...")

    # ... verify it can catch an uncaught exception ...
    raise ValueError("This is a test exception")

# -------------------------------------------------------------------------------------------------

def test_log_configuration():
    """
    Test the LogConfiguration class.
    """
    with patch('logging.FileHandler') as mock_file_handler:
        log_config = LogConfiguration(with_console_logging=True,
                                      with_file_logging=True,
                                      log_file_path='logs/svlearn.log')
        log_config.set_level()

        logger = logging.getLogger('SUPPORTVECTORS-LLM-BOOTCAMP')
        logger.info("Application is starting...")

        mock_file_handler.assert_called_once_with('logs/svlearn.log')

# -------------------------------------------------------------------------------------------------

def test_log_configuration_without_file_logging():
    """
    Test the LogConfiguration class without file logging.
    """
    with patch('logging.FileHandler') as mock_file_handler:
        log_config = LogConfiguration(with_console_logging=True,
                                      with_file_logging=False,
                                      log_file_path=None)
        log_config.set_level()

        logger = logging.getLogger('SUPPORTVECTORS-LLM-BOOTCAMP')
        logger.info("Application is starting...")

        mock_file_handler.assert_not_called()

