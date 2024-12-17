from svlearn.config.configuration import ConfigurationMixin
from enum import Enum

config = ConfigurationMixin().load_config()
current_task = config['current_task']

#  -------------------------------------------------------------------------------------------------
#  Enums
#  -------------------------------------------------------------------------------------------------
class Task(Enum):
    MNIST = 'mnist-classification'
    TREE = 'tree-classification'


__all__ = ['config', 'current_task', 'Task']