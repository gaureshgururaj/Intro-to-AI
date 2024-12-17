#  Copyright (c) 2020.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
#
import os
from pathlib import Path
import pytest
from unittest.mock import patch

from .svexception import UnspecifiedDirectoryError, UnspecifiedFileError


# -----------------------------------------------------------------------------
def directory_exists(dir_name: str = None) -> bool:
    """
    Checks if a given directory exists in the filesystem
    
    Example:
        ```python
        if directory_exists('data'):
            print('Directory exists')
        ```

    Parameters:
        dir_name: name of the directory as a string

    Returns:
        True if the directory exists, False otherwise.
    """
    if not dir_name:
        raise UnspecifiedDirectoryError('dir_name')
    path = Path(dir_name)
    return path.exists() and path.is_dir()


# -----------------------------------------------------------------------------
def directory_readable(dir_name: str = None) -> bool:
    """
    Checks if a given directory exists and its contents are readable.
    
    Example:
        ```python
        if directory_readable('data'):
            print('Directory is readable')
        ```

    Parameters:
        dir_name: name of the directory as a string

    Returns:
        True if the directory exists and its contents are readable, False otherwise.
    """
    if not dir_name:
        raise UnspecifiedDirectoryError('dir_name')
    path: Path = Path(dir_name)
    return path.exists() and path.is_dir() and os.access(path, os.R_OK)


# -----------------------------------------------------------------------------
def directory_writable(dir_name: str = None) -> bool:
    """
    Checks if a given directory exists and writable into
    
    Example:
        ```python
        if directory_writable('data'):
            print('Directory is writable')
        ```

    Parameters:
        dir_name: name of the directory as a string

    Returns:
        True if the directory exists and is writable, False otherwise.
    """
    if not dir_name:
        raise UnspecifiedDirectoryError('dir_name')
    path: Path = Path(dir_name)
    return path.exists() and os.access(path, os.W_OK)


# -----------------------------------------------------------------------------
def directory_is_empty(dir_name: str = None) -> bool:
    """
    Chwcks if a given directory exists and is empty.

    Example:
        ```python
        if directory_is_empty('data'):
            print('Directory is empty')
        ```

    Parameters:
        dir_name: name of the directory as a string

    Returns:
        True if the directory exists and is empty, False otherwise.
    """
    if not dir_name:
        raise UnspecifiedDirectoryError('dir_name')
    path: Path = Path(dir_name)
    return path.exists() and path.is_dir() and not path.iterdir()


# -----------------------------------------------------------------------------
def ensure_directory(dir_name: str = None) -> None:
    """
    Ensures that a given directory exists in the filesystem. If it does not exist, it creates it.

    Example:
        ```python
        ensure_directory('data')
        ```

    Parameters:
        dir_name: name of the directory as a string

    Returns:
        None
    """
    if not dir_name:
        raise UnspecifiedDirectoryError('dir_name')
    path: Path = Path(dir_name)
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
def file_exists(file_name: str) -> bool:
    """
    Checks if a given file exists in the filesystem
    
    Example:
        ```python
        if file_exists('data.csv'):
            print('File exists')
        ```

    Parameters:
        file_name: name of the file as a string

    Returns:
        True if the file exists, False otherwise.
    """
    if not file_name:
        raise UnspecifiedFileError('file_name')
    path = Path(file_name)
    return path.exists() and path.is_file()


def delete_file(file_name: str) -> None:
    """
     Checks if a given file exists in the filesystem, and delete
     it if it does.

    Example:
        ```python
        delete_file('data.csv')
        ```

    Parameters:
        file_name: name of the file as a string

    Returns:
        None
     """
    if not file_name:
        raise UnspecifiedFileError('file_name')
    Path(file_name).unlink(missing_ok=True)


def file_is_empty(file_name: str = None) -> bool:
    """
    Checks if a given file exists in the filesystem
    
    Example:
        ```python
        if file_is_empty('data.csv'):
            print('File is empty')
        ```

    Parameters:
        file_name: name of the file as a string

    Returns:
        True if the file exists and is empty, False otherwise.
    """
    if not file_name:
        raise UnspecifiedFileError('file_name')
    path = Path(file_name)
    # there seems to be no way yet to directly use pathlib
    return path.exists() and path.is_file() and os.stat(file_name).st_size == 0


def file_not_empty(file_name: str = None) -> bool:
    """
    Checks if a given file exists in the filesystem
    
    Example:
        ```python
        if file_not_empty('data.csv'):
            print('File is not empty')
        ```

    Parameters:
        file_name: name of the file as a string

    Returns:
        True if the file exists and is not empty, False otherwise.
    """
    return not file_is_empty(file_name)


# -----------------------------------------------------------------------------
def file_readable(file_name: str = None) -> bool:
    """
    Checks if a given file exists and its contents are readable.
    
    Example:
        ```python
        if file_readable('data.csv'):
            print('File is readable')
        ```

    Parameters:
        file_name: name of the file as a string

    Returns:
        True if the file exists and its contents are readable, False otherwise.
    """
    if not file_name:
        raise UnspecifiedFileError('file_name')
    path: Path = Path(file_name)
    return path.exists() and path.is_file() and os.access(path, os.R_OK)


# -----------------------------------------------------------------------------
def file_writable(file_name: str = None) -> bool:
    """
    Checks if a given file exists and its contents are readable.
    
    Example:
        ```python
        if file_writable('data.csv'):
            print('File is writable')
        ```

    Parameters:
        file_name: name of the file as a string

    Returns:
        True if the file exists and its contents are writable, False otherwise.
    """
    if not file_name:
        raise UnspecifiedFileError('file_name')
    path: Path = Path(file_name)
    return path.exists() and path.is_file() and os.access(path, os.W_OK)

# -----------------------------------------------------------------------------

def check_valid_file (path:str) -> None:
    """
    Checks if the given path points to an existing, readable, non-empty file
    
    Example:
        ```python
        check_valid_file('data.csv')
        ```

    Parameters:
        path: the path to the file

    Returns:
        None
    """
    valid_path: bool = file_exists(path) and file_readable(path) and file_not_empty(path)

    if not file_exists(path):
        errorMsg = f'Path does not point to an existing file: {path}'
        raise FileNotFoundError(errorMsg)
    if not file_readable(path):
        errorMsg = f'Path does not point to a readable file: {path}'
        raise FileNotFoundError(errorMsg)
    if  file_is_empty(path):
        errorMsg = f'Path does not point to an empty file: {path}'
        raise FileNotFoundError(errorMsg)  

# -----------------------------------------------------------------------------

def check_valid_directory (path:str) -> None:
    """
    Checks if the given path points to an existing, readable, non-empty directory
    
    Example:
        ```python
        check_valid_directory('data')
        ```

    Parameters:
        path: the path to the directory

    Returns:
        None
    """
    valid_path: bool = directory_exists(path) and directory_readable(path) and directory_is_empty(path)

    if not directory_exists(path):
        errorMsg = f'Path does not point to an existing directory: {path}'
        raise FileNotFoundError(errorMsg)
    if not directory_readable(path):
        errorMsg = f'Path does not point to a readable directory: {path}'
        raise FileNotFoundError(errorMsg)
    if  not directory_is_empty(path):
        errorMsg = f'Path does not point to an empty directory: {path}'
        raise FileNotFoundError(errorMsg)
    


