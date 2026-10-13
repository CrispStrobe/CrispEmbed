# Ryu shortest double conversion

Unmodified double-to-shortest sources from
[ulfjack/ryu](https://github.com/ulfjack/ryu) at
`4c0618b0e44f7ef027ebae05d2cc7812048f7c8f`, by Ulf Adams and contributors.
Distributed under the Apache-2.0 option of its dual license; see LICENSE-Apache2
and the retained source notices.

The small wrapper in `src/core/shortest_double.c` prefixes the three public
double-conversion symbols. GemmaDecision adapts scientific exponent spelling and
the fixed/scientific threshold to Python's `str(float)`. This avoids raising the
project's macOS 12 / iOS 15 deployment targets for floating-point `std::to_chars`.
Model-free checks compare boundary values and fixed-seed IEEE samples with actual
Python string conversion. No other model's inference arithmetic changes.
