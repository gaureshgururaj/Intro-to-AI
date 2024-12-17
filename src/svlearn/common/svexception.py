#  Copyright (c) 2020.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
#
import pytest
from unittest.mock import patch

class SVError(Exception):
    """
    Error Constructor

    Example:
        ```python
        try:
            raise SVError('This is an error message')
        except SVError as e:
            print(e.message)
        ```
    
    Parameters:
        message (str): The error message
    """  
    def __init__(self, message: str=None):     
        self.message = message


# -----------------------------------------------------------------

class MissingArgumentError(SVError):
    """
    Exception raised when a required argument is missing

    Example:
        ```python
        try:
            raise MissingArgumentError('model')
        except MissingArgumentError as e:
            print(e.message)
        ```

    Parameters:
        arg (str): the name of the argument that is missing
    """
    def __init__(self, arg: str):
        self.message = f'A requirement argument: {arg} is missing!'


# -----------------------------------------------------------------

class UnspecifiedDirectoryError(SVError):
    """
    Exception raised when a required directory is not specified

    Example:
        ```python
        try:
            raise UnspecifiedDirectoryError(arg='xyzdir')
        except UnspecifiedDirectoryError as e:
            print(e.message)
        ```

    Parameters:
        arg (str): the name of the argument that should contain
                the directory name
        message (str): explanation of the error
    """
    def __init__(self, arg: str, message: str=None):
        super.__init__(message)
        self.arg = arg
        if not self.message:
            self.message = f'Directory name must be specified for the arg: {self.arg}'


# -----------------------------------------------------------------

class UnspecifiedFileError(SVError):
    """
    Exception raised when a required file is not specified

    Example:
        ```python
        try:
            raise UnspecifiedFileError(arg='xyzfile')
        except UnspecifiedFileError as e:
            print(e.message)
        ```

    Parameters:
        arg (str): the name of the argument that should contain
                the file name
        message (str): explanation of the error
    """
    def __init__(self, arg: str, message: str=None):
        super.__init__(message)
        self.arg = arg
        if not self.message:
            self.message = f'File name must be specified for the arg: {self.arg}'



# -----------------------------------------------------------------


if __name__ == "__main__":
    try:
        raise UnspecifiedDirectoryError(arg='xyzdir')
    except UnspecifiedDirectoryError as e:
        print(e.message)


# -----------------------------------------------------------------

def test_sverror():
    try:
        raise SVError('This is an error message')
    except SVError as e:
        assert e.message == 'This is an error message'

def test_missing_argument_error():
    try:
        raise MissingArgumentError('model')
    except MissingArgumentError as e:
        assert e.message == 'A requirement argument: model is missing!'

def test_unspecified_directory_error():
    try:
        raise UnspecifiedDirectoryError(arg='xyzdir')
    except UnspecifiedDirectoryError as e:
        assert e.message == 'Directory name must be specified for the arg: xyzdir'

def test_unspecified_file_error():
    try:
        raise UnspecifiedFileError(arg='xyzfile')
    except UnspecifiedFileError as e:
        assert e.message == 'File name must be specified for the arg: xyzfile'

