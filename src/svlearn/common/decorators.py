#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2023.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: Asif Qamar
#  -------------------------------------------------------------------------------------------------

from time import sleep
from typing import Any
from decorator import decorator
import logging as logger
from rich import print as rprint
import pytest
from unittest.mock import patch

# ------------------------------------------------------------------------------------------------------------------
def singleton(cls) -> Any:
    """
    Decorator that makes a class follow the Singleton design pattern.
    In other words, there can be at-most one object instance of the class,
    and the repeated call to the constructor will yield the same object instance.

    Example: 
        ```python
        @singleton
        class MyClass:
            pass
        ```

    Args:
        cls (class): The class to be decorated.

    Returns:
        class (Any): The decorated class.
    """
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance
# ------------------------------------------------------------------------------------------------------------------
@decorator
def log_time_taken(func, *args: tuple, **kwargs: dict[str, Any]):
    """
    Decorator that logs the time taken by the decorated function.

    Example:
        ```python
        @log_time_taken
        def test_function():
            sleep(2)
            print("Hello, World!")
        ```

    Args:
        func (function): The function to be decorated.
        *args: The positional arguments to the function.
        **kwargs: The keyword arguments to the function.

    Returns:
        Any: The result of the decorated function.
    """
    import time
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    border = "-" * 100 
    message ='\n'.join ([border, 
                         f" Time taken by {func.__name__}: {end - start} seconds", 
                         border])
    logger.info(message)
    rprint(message)
    return result

# ------------------------------------------------------------------------------------------------------------------

@log_time_taken
def test_function():
    sleep(2)
    rprint("Hello, World!")

if __name__ == "__main__":
    test_function()

# ------------------------------------------------------------------------------------------------------------------

# Command to run the test:
# python -m pytest src/svlearn/tests/common/decorators.py

# Test the singleton decorator
def test_singleton():
    @singleton
    class MyClass:
        pass
    
    instance1 = MyClass()
    instance2 = MyClass()
    try:
        assert instance1 is instance2, "Singleton pattern is not implemented correctly"
        print("Singleton pattern is implemented correctly")
    except AssertionError:
        print("Singleton pattern is not implemented correctly")

# Test the log_time_taken decorator
@patch('svlearn.common.decorators.logger.info')
@patch('svlearn.common.decorators.rprint')
def test_log_time_taken(mock_rprint, mock_logger_info):
    @log_time_taken
    def dummy_function():
        return "Executed"

    result = dummy_function()
    try:
        assert result == "Executed", "The decorated function did not return the expected result"
        print("The decorated function returned the expected result")
    except AssertionError:
        print("The decorated function did not return the expected result")
    assert mock_rprint.called, "The rprint function was not called"
    log_message = mock_logger_info.call_args[0][0]
    try:
        assert "Time taken by dummy_function" in log_message, "Time logging message format is incorrect"
        print("Time logging message format is correct")
    except AssertionError:
        print("Time logging message format is incorrect")