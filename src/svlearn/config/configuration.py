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

import logging as log
import os
from pathlib import Path

import ruamel.yaml as yaml
from ruamel.yaml import CommentedMap

from svlearn import CONFIG_PATH
from svlearn.common import SVError, file_exists

# Rick will help us get sensible stack-traces for debugging.
from rich.traceback import install
from rich.console import Console
install(show_locals=True)
console = Console()




ROOT_DIR = os.getenv("AI_INTRO_ROOT_DIR")
CONFIG_YAML = 'ai-intro-config.yaml'


class ConfigurationMixin:
    """
    A mixin class to load configuration from a YAML file
    """
    def load_config(self, config_file: Path = None) -> CommentedMap:
        """
        Loads a configuration from a YAML file
        
        Example:
            ```python
            config = load_config('config.yaml')
            ```

        Parameters:
            config_file: The path to the configuration file

        Returns:
            The configuration as a dictionary
        """
        default_config_file_dir = ROOT_DIR
        if default_config_file_dir is None:
            default_config_file_dir = str(Path.cwd())

        print(default_config_file_dir)
        if config_file is None:
            log.warning('No configuration file specified. Trying the default location')
            # We are going to try the default location
            config_file = CONFIG_PATH
            log.warning(f'Loading configuration from {config_file} if it exists')

        if not file_exists(config_file):
            error_msg = f'Configuration file not found: {config_file}'
            log.error(error_msg)
            raise SVError(error_msg)
        else:
            log.info(f'Configuration file found: {config_file}')

        log.info(f'Loading configuration from {config_file}')
        loader = yaml.YAML()

        try:
            with open(config_file, 'r') as config_file:
                config = loader.load(config_file)
                log.info(f'Configuration loaded from {config_file}')

        except FileNotFoundError:
            log.error(f'Configuration file not found: {config_file}')
            raise SVError(f'Configuration file not found: {config_file}')
        except Exception as e:
            log.error(f'Error loading configuration from {config_file}: {e}')
            raise SVError(f'Error loading configuration from {config_file}: {e}')

        return config


if __name__ == '__main__':
    # before running this, make sure the cwd() is set to the project root.
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    print(config['database'])
    variant_ = config["database"]["variant"]
    print(f'---{variant_}---')
