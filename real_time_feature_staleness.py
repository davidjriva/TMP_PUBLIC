'''
Real-Time Feature Staleness (Graph Traversal / Logic)
In a feature store, some data is "fresh" and some is "stale." You need to decide whether to recompute a node in a DAG.

The Problem:
You are given a Directed Acyclic Graph (DAG) where each node is a "Feature." Each node has:

A last_updated timestamp.

A max_staleness (how long the value is valid).

A compute_fn that takes its dependencies as input.

The Task:
Write a function get_feature(node_name) that:

Checks if the feature is "fresh" (current time - last_updated < max_staleness).

If fresh, return the stored value.

If stale, recursively check/compute all its dependencies, then run compute_fn, and update the timestamp.
'''