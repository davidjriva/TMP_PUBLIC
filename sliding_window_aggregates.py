'''
Sliding Window Aggregates (Algorithms / Data Structures)
Real-time ML often requires calculating "The number of logins in the last 24 hours" for every incoming event.

The Problem: Design a data structure FeatureWindow that supports two operations:
    add_event(timestamp, value): Adds a numerical value at a specific time.
    get_sum(current_time, window_size): Returns the sum of all values where timestamp > (current_time - window_size).
    
Optimization Goal:
    add_event should be O(1) or $O(log N)

.get_sum must be faster than O(N)(scanning all events).

Bonus: How would you handle this if you have 10 million events and need to keep 
memory usage low? (Hint: Think about bucketing or a Circular Buffer/Deque).
'''

from datetime import datetime, timedelta
from bisect import insort, bisect_left, bisect_right
from collections import deque, defaultdict

'''
    {
        (year,day,hour): [[list of (timestamp, values)], running sum of the list of values]
    }
'''



class FeatureWindow:
    def __init__(self):
        def _make_default():
            return [[], 0]

        self.buckets = defaultdict(_make_default)

    def add_event(self, timestamp: datetime, value: float):
        key = (timestamp.year, timestamp.timetuple().tm_yday, timestamp.hour)

        if not self.buckets[key][0] or timestamp >= self.buckets[key][0][-1][0]:
            self.buckets[key][0].append((timestamp, value))
        else:
            insort(self.buckets[key][0], (timestamp, value))

        self.buckets[key][1] += value

    def get_sum(self, current_time: datetime, window_size: int) -> float:
        '''
            window_size (int): The number of hours back to look
        '''

        # print(f"get_sum({current_time}, {window_size})")

        cutoff = current_time - timedelta(hours=window_size)

        total = 0

        hours_back = window_size

        for i in range(hours_back + 1):
            dt = current_time - timedelta(hours=i)
            key = (dt.year, dt.timetuple().tm_yday, dt.hour)
            
            # Filter events in this hour
            events, hour_sum = self.buckets[key]

            if not events:
                continue # skip empty buckets
            
            # Case 1: The window starts and ends in this SAME bucket
            if window_size == 0 or (dt.hour == cutoff.hour and dt.hour == current_time.hour):
                # Only sum events between cutoff and current_time
                idx_start = bisect_left(events, cutoff, key=lambda x: x[0])
                idx_end = bisect_right(events, current_time, key=lambda x: x[0])
                total += sum(v for t, v in events[idx_start:idx_end])
            elif i == 0:
                # Find events occurring AFTER the cutoff
                idx = bisect_right(events, current_time, key=lambda x: x[0])
                total += sum(v for t, v in events[:idx])
            elif i == hours_back:  # first hour in window, may need partial sum
                idx = bisect_left(events, cutoff, key=lambda x: x[0])
                total += sum(v for t,v in events[idx:])
            else:
                total += hour_sum

        return total



if __name__ == "__main__":
    # Mock events: (timestamp, transaction_value)
    # We'll use a fixed 'current_time' of 2024-01-01 12:00:00 for testing
    FW = FeatureWindow()

    now = datetime(2024, 1, 1, 12, 0, 0)

    events = [
        # Events well outside a 1-hour window (Older than 11:00 AM)
        (datetime(2024, 1, 1, 9, 30), 100.0),
        (datetime(2024, 1, 1, 10, 45), 50.0),
        
        # Events just inside a 1-hour window (Between 11:00 AM and 12:00 PM)
        (datetime(2024, 1, 1, 11, 0, 5), 20.0),   # 59 mins ago
        (datetime(2024, 1, 1, 11, 30), 10.0),     # 30 mins ago
        (datetime(2024, 1, 1, 11, 55), 15.0),     # 5 mins ago
        
        # "Future" event (relative to now - should usually be ignored or handled)
        (datetime(2024, 1, 1, 12, 0, 5), 500.0),
    ]

    expected = [0.0,0.0,20.0,30.0,45.0,45.0]

    for timestamp, value in events:
        FW.add_event(timestamp, value)
        # print(FW.get_sum(now, 1))
        assert FW.get_sum(now, 1) == expected.pop(0), f"Expected {expected[0]} but got {FW.get_sum(now, 1)}"

    # Example Test Case for the Implementation:
    # window_size = 3600 (1 hour in seconds)
    # Expected Sum: 20.0 + 10.0 + 15.0 = 45.0