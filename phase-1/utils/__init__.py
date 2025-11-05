"""
Utility functions for the segmentation pipeline
"""
from .logger import setup_logger
from .system import get_num_workers

__all__ = ['setup_logger', 'get_num_workers']
