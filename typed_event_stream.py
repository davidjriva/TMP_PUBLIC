'''
The "Typed" Event Stream (Data Structures / Logic)
Chalk processes streams of events that must be validated against a schema in real-time.

The Problem:
Write a SchemaValidator that takes a nested dictionary (the event) and a schema definition. The schema can contain primitive types (int, str), nested schemas (dictionaries), or a special Optional[T] type.

Constraints:

If a key is missing and not Optional, return a list of missing fields.

If a type is incorrect, return the path to the error (e.g., "user.settings.theme").

Recursive Challenge: Handle "Circular Schemas" (e.g., a User has a Friend, who is also a User).

Why this fits: This tests your ability to navigate nested structures recursively and handle Python's type system, similar to the @features decorator work you've done.
'''