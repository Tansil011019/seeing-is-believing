"""
System utilities for resource management
"""
from multiprocessing import cpu_count


def get_num_workers(num_workers: int) -> int:
    """
    Get the actual number of workers to use
    
    Args:
        num_workers: Requested number of workers (-1 for all cores)
        
    Returns:
        Actual number of workers to use
    """
    if num_workers == -1:
        return cpu_count()
    return num_workers


def format_worker_info(num_workers: int) -> str:
    """
    Format worker information for logging
    
    Args:
        num_workers: Number of workers being used
        
    Returns:
        Formatted string describing worker usage
    """
    if num_workers == cpu_count():
        return f"{num_workers} (using all available CPU cores)"
    return f"{num_workers} (using specified number)"
