"""
Assignment: Divide & Conquer + Greedy Algorithms
Task 1: Find min/max using divide and conquer
Task 2: 3D Printer Queue Optimization using greedy batching
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ============================================================
# TASK 1: Divide & Conquer - Min/Max
# ============================================================

def find_min_max(arr: List[float]) -> Tuple[float, float]:
    """
    Find minimum and maximum elements in an array using divide and conquer.
    
    Uses recursive divide-and-conquer approach with O(n) time complexity.
    
    Args:
        arr: List of numbers (can be integers or floats)
    
    Returns:
        Tuple of (min_value, max_value)
    
    Raises:
        ValueError: If input array is empty
    
    Examples:
        >>> find_min_max([3, 1, 4, 1, 5])
        (1, 5)
        >>> find_min_max([42])
        (42, 42)
        >>> find_min_max([-5, 0, 10])
        (-5, 10)
    """
    if not arr:
        raise ValueError("Input array cannot be empty")
    
    def solve(left: int, right: int) -> Tuple[float, float]:
        """
        Recursive helper function to find min/max in arr[left:right+1].
        
        Args:
            left: Left index (inclusive)
            right: Right index (inclusive)
        
        Returns:
            Tuple of (min_value, max_value) for the subarray
        """
        # Base case: single element
        if left == right:
            return (arr[left], arr[left])
        
        # Base case: two elements
        if right - left == 1:
            if arr[left] < arr[right]:
                return (arr[left], arr[right])
            else:
                return (arr[right], arr[left])
        
        # Divide: split in the middle
        mid = (left + right) // 2
        
        # Conquer: solve left and right halves
        left_min, left_max = solve(left, mid)
        right_min, right_max = solve(mid + 1, right)
        
        # Combine: merge results with 2 comparisons
        overall_min = left_min if left_min < right_min else right_min
        overall_max = left_max if left_max > right_max else right_max
        
        return (overall_min, overall_max)
    
    return solve(0, len(arr) - 1)


# ============================================================
# TASK 2: Greedy Scheduling - 3D Printer Queue Optimization
# ============================================================

@dataclass
class PrintJob:
    """Represents a single 3D printing job."""
    id: str
    volume: float
    priority: int  # 1 (highest) to 3 (lowest)
    print_time: int  # in minutes


@dataclass
class PrinterConstraints:
    """Constraints for the 3D printer."""
    max_volume: float
    max_items: int


def optimize_printing(
    print_jobs: List[Dict],
    constraints: Dict
) -> Dict:
    """
    Optimize 3D printing queue using greedy batching algorithm.
    
    Groups print jobs into batches respecting priority ordering and printer constraints.
    Higher priority jobs (priority 1) are scheduled before lower priority jobs (priority 3).
    Jobs can be batched together if they fit within max_volume and max_items constraints.
    
    Args:
        print_jobs: List of dictionaries with keys: id, volume, priority, print_time
        constraints: Dictionary with keys: max_volume, max_items
    
    Returns:
        Dictionary with keys:
            - "print_order": List of job IDs in the order they will be printed
            - "total_time": Total printing time in minutes
    
    Raises:
        ValueError: If constraints are invalid or a job cannot fit in any batch
    """
    # Validate constraints
    if constraints["max_volume"] <= 0:
        raise ValueError("max_volume must be positive")
    if constraints["max_items"] < 1:
        raise ValueError("max_items must be at least 1")
    
    # Convert constraints to dataclass
    printer_constraints = PrinterConstraints(
        max_volume=constraints["max_volume"],
        max_items=constraints["max_items"]
    )
    
    # Convert and validate print jobs
    jobs: List[PrintJob] = []
    for job_dict in print_jobs:
        # Validate required fields
        if not job_dict.get("id"):
            raise ValueError("Job id cannot be empty")
        if job_dict.get("volume", 0) <= 0:
            raise ValueError(f"Job {job_dict.get('id')}: volume must be positive")
        if job_dict.get("print_time", 0) <= 0:
            raise ValueError(f"Job {job_dict.get('id')}: print_time must be positive")
        if job_dict.get("priority") not in [1, 2, 3]:
            raise ValueError(f"Job {job_dict.get('id')}: priority must be 1, 2, or 3")
        
        # Check if single job exceeds constraints
        if job_dict["volume"] > printer_constraints.max_volume:
            raise ValueError(
                f"Job {job_dict['id']} volume {job_dict['volume']} exceeds "
                f"max_volume {printer_constraints.max_volume}"
            )
        
        jobs.append(PrintJob(
            id=job_dict["id"],
            volume=job_dict["volume"],
            priority=job_dict["priority"],
            print_time=job_dict["print_time"]
        ))
    
    # Organize jobs by priority, preserving input order
    priority_queues: Dict[int, List[PrintJob]] = {1: [], 2: [], 3: []}
    for job in jobs:
        priority_queues[job.priority].append(job)
    
    # Track which jobs have been processed
    processed = set()
    print_order: List[str] = []
    total_time = 0
    
    # Process jobs in priority order (1 -> 2 -> 3)
    while len(processed) < len(jobs):
        # Find the highest priority queue with remaining jobs
        anchor_priority = None
        anchor_job = None
        
        for priority in [1, 2, 3]:
            for job in priority_queues[priority]:
                if job.id not in processed:
                    anchor_priority = priority
                    anchor_job = job
                    break
            if anchor_job is not None:
                break
        
        if anchor_job is None:
            break
        
        # Start a new batch with the anchor job
        batch: List[PrintJob] = [anchor_job]
        batch_volume = anchor_job.volume
        batch_max_time = anchor_job.print_time
        
        processed.add(anchor_job.id)
        
        # Try to fill the batch
        # First, try to add jobs from the same priority queue
        for job in priority_queues[anchor_priority]:
            if job.id in processed:
                continue
            
            # Check if adding this job would violate constraints
            if (batch_volume + job.volume <= printer_constraints.max_volume and
                len(batch) < printer_constraints.max_items):
                batch.append(job)
                batch_volume += job.volume
                batch_max_time = max(batch_max_time, job.print_time)
                processed.add(job.id)
        
        # Then, try to add jobs from lower priority queues (priority+1, priority+2)
        # but only if they don't delay any higher priority jobs
        for lower_priority in [anchor_priority + 1, anchor_priority + 2]:
            if lower_priority > 3:
                break
            
            for job in priority_queues[lower_priority]:
                if job.id in processed:
                    continue
                
                # Check if adding this job would violate constraints
                if (batch_volume + job.volume <= printer_constraints.max_volume and
                    len(batch) < printer_constraints.max_items):
                    batch.append(job)
                    batch_volume += job.volume
                    batch_max_time = max(batch_max_time, job.print_time)
                    processed.add(job.id)
        
        # Close the batch: add all job IDs to print_order
        for job in batch:
            print_order.append(job.id)
        
        # Add batch time to total
        total_time += batch_max_time
    
    return {
        "print_order": print_order,
        "total_time": total_time
    }


# ============================================================
# TEST FUNCTIONS
# ============================================================

def test_min_max():
    """Test cases for find_min_max function."""
    print("Testing find_min_max...")
    
    # Test 1: Basic case
    assert find_min_max([3, 1, 4, 1, 5]) == (1, 5)
    print("✓ Test 1 passed")
    
    # Test 2: Single element
    assert find_min_max([42]) == (42, 42)
    print("✓ Test 2 passed")
    
    # Test 3: Two elements
    assert find_min_max([5, 3]) == (3, 5)
    print("✓ Test 3 passed")
    
    # Test 4: Negative numbers
    assert find_min_max([-5, 0, 10, -3]) == (-5, 10)
    print("✓ Test 4 passed")
    
    # Test 5: Empty list (should raise ValueError)
    try:
        find_min_max([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()
        print("✓ Test 5 passed (empty list raises ValueError)")
    
    # Test 6: All same values
    assert find_min_max([7, 7, 7, 7]) == (7, 7)
    print("✓ Test 6 passed")
    
    print("All min_max tests passed!\n")


def test_printing_optimization():
    """Test cases for optimize_printing function."""
    print("Testing optimize_printing...")
    
    # Test 1: Same priority
    jobs1 = [
        {"id": "M1", "volume": 100, "priority": 1, "print_time": 120},
        {"id": "M2", "volume": 150, "priority": 1, "print_time": 90},
        {"id": "M3", "volume": 200, "priority": 1, "print_time": 150}
    ]
    constraints1 = {"max_volume": 300, "max_items": 2}
    result1 = optimize_printing(jobs1, constraints1)
    
    print(f"Test 1 result: {result1}")
    print(f"  Expected total_time: 270")
    print(f"  Got total_time: {result1['total_time']}")
    assert result1["total_time"] == 270, f"Expected 270, got {result1['total_time']}"
    assert len(result1["print_order"]) == 3
    print("✓ Test 1 passed")
    
    # Test 2: Different priorities
    jobs2 = [
        {"id": "M1", "volume": 100, "priority": 2, "print_time": 120},
        {"id": "M2", "volume": 150, "priority": 1, "print_time": 90},
        {"id": "M3", "volume": 200, "priority": 3, "print_time": 150}
    ]
    constraints2 = {"max_volume": 300, "max_items": 2}
    result2 = optimize_printing(jobs2, constraints2)
    
    print(f"Test 2 result: {result2}")
    print(f"  Expected total_time: 270")
    print(f"  Got total_time: {result2['total_time']}")
    assert result2["total_time"] == 270, f"Expected 270, got {result2['total_time']}"
    assert len(result2["print_order"]) == 3
    # M2 (priority 1) should come before M1 and M3
    assert result2["print_order"][0] == "M2"
    print("✓ Test 2 passed")
    
    # Test 3: Exceeding printer limits
    jobs3 = [
        {"id": "M1", "volume": 250, "priority": 1, "print_time": 180},
        {"id": "M2", "volume": 200, "priority": 1, "print_time": 150},
        {"id": "M3", "volume": 180, "priority": 2, "print_time": 120}
    ]
    constraints3 = {"max_volume": 300, "max_items": 2}
    result3 = optimize_printing(jobs3, constraints3)
    
    print(f"Test 3 result: {result3}")
    print(f"  Expected total_time: 450")
    print(f"  Got total_time: {result3['total_time']}")
    assert result3["total_time"] == 450, f"Expected 450, got {result3['total_time']}"
    assert len(result3["print_order"]) == 3
    print("✓ Test 3 passed")
    
    print("All printing optimization tests passed!\n")


if __name__ == "__main__":
    test_min_max()
    test_printing_optimization()
    print("=" * 50)
    print("All tests completed successfully!")
