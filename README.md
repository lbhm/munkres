# Munkres

This repository contains a fork of the `munkres` packages by Brian M. Clapper.
The original repository can be found at <https://github.com/bmc/munkres>.
The fork is (mostly) API-compatible with the original package, so the documentation at
<https://software.clapper.org/munkres/api/index.html> still applies.

> [!IMPORTANT]
> This fork ported the implementation to Numpy and does not support irregular matrices.

## Introduction

The Munkres module provides an O(n^3) implementation of the Munkres algorithm
(also called the [Hungarian algorithm][] or the Kuhn-Munkres algorithm).
The algorithm models an assignment problem as an NxM cost matrix, where
each element represents the cost of assigning the ith worker to the jth
job, and it figures out the least-cost solution, choosing a single item
from each row and column in the matrix, such that no row and no column are
used more than once.

This particular implementation is based on
<https://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html>.

[Hungarian algorithm]: https://en.wikipedia.org/wiki/Hungarian_algorithm

See the docs on the [project page][] for more details.

[project page]: https://software.clapper.org/munkres/

## Installation

The fork is not published on PyPI. You can install it directly from the repository
using `pip` or `uv`:

```bash
pip install git+https://github.com/lbhm/munkres
# or
uv pip install git+https://github.com/lbhm/munkres
```

## Copyright

&copy; 2008-2019 Brian M. Clapper

&copy; 2025 Lennart Behme

## License

Licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE.md) for details.
