#  Copyright (c) 2020.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
#

import logging
from pathlib import Path
import pytest
from unittest.mock import patch

import torch  # pytorch library

from svlearn.common.svexception import SVError
from svlearn.common.utils import file_readable, UnspecifiedFileError


# -----------------------------------------------------------------
def save_model(state_dict: dict, file_path: str) -> None:
    """
    Saves the model to the file-system

    Example:
        ```python
        save_model(model.state_dict(), 'models/model.pth')
        ```

    Parameters:
        state_dict: The state-dictionary of the model
        file_path: The full-path to the file to store it in

    Returns:
        None
    """
    if not file_path:
        raise UnspecifiedFileError("file_path")
    path: Path = Path(file_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving the model state-dictionary to the file: {file_path}")
    torch.save(state_dict, file_path)


# -----------------------------------------------------------------
def load_model(file_path: str) -> dict:
    """
    Save the model to the file-system

    Example:
        ```python
        state_dict = load_model('models/model.pth')
        ```

    Parameters:
        file_path: The full-path to the file to load the model from

    Returns:
        The state-dictionary of the model
    """
    # pre-conditions check.
    if not file_path:
        raise UnspecifiedFileError("file_path")
    if not file_readable(file_path):
        raise SVError(f"The file path specified is not readable: {file_path}")

    logging.info("Loading the model state-dictionary from the file: {file_path}")
    return torch.load(file_path)

# -----------------------------------------------------------------

def test_save_model():
    """
    Test the save_model function
    """
    with patch("torch.save") as mock_torch_save:
        save_model({}, "models/model.pth")
        mock_torch_save.assert_called_once()


def test_load_model():
    """
    Test the load_model function
    """
    with patch("torch.load") as mock_torch_load:
        load_model("models/model.pth")
        mock_torch_load.assert_called_once()
