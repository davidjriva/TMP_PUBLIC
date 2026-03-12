'''
Transactional Feature Aggregation (Data Structures / Concurrency)
Chalk needs to aggregate data (like "sum of transfers") as events stream in.

The Problem:
Implement a TransactionTracker that supports:

record(user_id, amount, timestamp): Stores a transaction.

get_velocity(user_id, window_seconds): Returns the sum of transactions for that user in the last X seconds.

Constraint:

You must handle out-of-order events (e.g., an event from 10:00 AM arrives at 10:05 AM).

Follow-up: How would you make this thread-safe in Python if multiple record calls happen simultaneously? (Discussing the Global Interpreter Lock (GIL) vs. fine-grained locking or Atomic counters is a huge plus).
'''