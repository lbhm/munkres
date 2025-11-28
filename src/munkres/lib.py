"""Munkres (Hungarian) algorithm implementation.

The Munkres module provides an implementation of the Munkres algorithm
(also called the Hungarian algorithm or the Kuhn-Munkres algorithm), useful for
solving the Assignment Problem.

For complete usage documentation, see https://software.clapper.org/munkres/.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

type IntMatrix = np.ndarray[tuple[int, int], np.dtype[np.int32]]
type Matrix = np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]


DISALLOWED = float("nan")
STARRED = 1
PRIMED = 2


class UnsolvableMatrixError(Exception):
    """Exception raised for unsolvable matrices."""


class Munkres:
    """Compute the Munkres solution to the classical assignment problem.

    See the module documentation for usage instructions and background.
    """

    C: Matrix
    n: int
    Z0_r: int
    Z0_c: int
    marked: IntMatrix
    path: IntMatrix
    row_covered: np.ndarray[tuple[int], np.dtype[np.bool_]]
    col_covered: np.ndarray[tuple[int], np.dtype[np.bool_]]

    def compute(self, cost_matrix: ArrayLike) -> list[tuple[int, int]]:
        """Return the indexes for the lowest-cost assignments.

        Returns a list of `(row, column)` tuples that can be used to traverse the
        matrix.

        **WARNING**: This code handles square and rectangular matrices. It
        does *not* handle irregular matrices.

        Args:
            cost_matrix (list of lists of numbers): The cost matrix. If this
                cost matrix is not square, it will be padded with zeros, via a call
                to `pad_matrix()`. (This method does *not* modify the caller's
                matrix. It operates on a copy of the matrix.)

        Returns:
            A list of `(row, column)` tuples that describe the lowest cost path
            through the matrix.
        """
        cost_matrix_arr: Matrix = np.asarray(cost_matrix)
        if cost_matrix_arr.ndim != 2:  # noqa: PLR2004
            raise ValueError("Input matrix must be 2D.")

        self.C = pad_matrix(cost_matrix_arr)
        self.n = len(self.C)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.full((self.n * 2, 2), 0, dtype=np.int32)
        self.marked = np.full((self.n, self.n), 0, dtype=np.int32)
        self.__reset_covers()

        done = False
        step = 1

        steps = {
            1: self.__step1,
            2: self.__step2,
            3: self.__step3,
            4: self.__step4,
            5: self.__step5,
            6: self.__step6,
        }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        shape = cost_matrix_arr.shape
        rows, cols = np.where(self.marked[: shape[0], : shape[1]] == STARRED)
        return list(zip(rows.tolist(), cols.tolist(), strict=True))

    def __step1(self) -> int:
        """Normalize rows by subtracting their minimum element."""
        # Detect rows that are entirely DISALLOWED (all NaN)
        all_nan = np.isnan(self.C).all(axis=1)
        if np.any(all_nan):
            first_bad = int(np.where(all_nan)[0][0])
            raise UnsolvableMatrixError(f"Row {first_bad} is entirely DISALLOWED.")

        # Compute row-wise minima ignoring NaNs and subtract via broadcasting.
        row_mins = np.nanmin(self.C, axis=1)  # shape (n,)
        # Subtract per-row minima; keep DISALLOWED (NaN) entries as NaN
        self.C = np.where(np.isnan(self.C), np.nan, self.C - row_mins[:, None])
        return 2

    def __step2(self) -> int:
        """Star uncovered zeros that have no starred zero in their row or column."""
        for i in range(self.n):
            for j in range(self.n):
                if (self.C[i, j] == 0) and (not self.col_covered[j]) and (not self.row_covered[i]):
                    self.marked[i, j] = STARRED
                    self.col_covered[j] = True
                    self.row_covered[i] = True
                    break

        self.__reset_covers()
        return 3

    def __step3(self) -> int:
        """Cover each column that contains a starred zero.

        If K columns are covered, the starred zeros describe a complete set of unique
        assignments. In this case, go to DONE; otherwise go to Step 4.
        """
        # Find columns with starred zeros using vectorized operation
        starred_cols = np.any(self.marked == STARRED, axis=0)
        self.col_covered[starred_cols & ~self.col_covered] = True
        count = np.sum(starred_cols)

        return 7 if count >= self.n else 4

    def __step4(self) -> int:
        """Prime uncovered zeros and adjust covers until none remain.

        If there is no starred zero in the row containing this primed zero, go to Step 5.
        Otherwise, cover this row and uncover the column containing the starred zero. Continue in
        this manner until there are no uncovered zeros left. Save the smallest uncovered value and
        go to Step 6.
        """
        step = 0
        done = False
        row = 0
        col = 0
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero(row, col)
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row, col] = PRIMED
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self) -> int:
        """Implement the alternating primed/starred zero path adjustment.

        Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count, 0] = self.Z0_r
        path[count, 1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(int(path[count, 1]))
            if row >= 0:
                count += 1
                path[count, 0] = row
                path[count, 1] = path[count - 1, 1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(int(path[count, 0]))
                count += 1
                path[count, 0] = path[count - 1, 0]
                path[count, 1] = col

        self.__convert_path(path, count)
        self.__reset_covers()
        self.__erase_primes()
        return 3

    def __step6(self) -> int:
        """Adjust uncovered columns and covered rows by the smallest uncovered value.

        Return to Step 4 without altering any stars, primes, or covered lines.
        """
        # Create broadcasting masks
        row_mask = self.row_covered[:, np.newaxis]  # Shape (n, 1)
        col_mask = self.col_covered[np.newaxis, :]  # Shape (1, n)
        valid_mask = ~np.isnan(self.C)

        minval = np.min(self.C[~row_mask & ~col_mask & valid_mask])

        # Track actual changes
        events = np.sum(valid_mask & row_mask, dtype=np.int32) + np.sum(
            valid_mask & ~col_mask, dtype=np.int32
        )
        events -= np.sum(valid_mask & row_mask & ~col_mask, dtype=np.int32) * 2

        if events == 0:
            raise UnsolvableMatrixError("Matrix cannot be solved!")

        # Vectorized matrix adjustment
        self.C[valid_mask & row_mask] += minval  # type: ignore[misc]
        self.C[valid_mask & ~col_mask] -= minval  # type: ignore[misc]

        return 4

    def __find_a_zero(self, i0: int = 0, j0: int = 0) -> tuple[int, int]:
        """Find the first uncovered element with value 0."""
        row = -1
        col = -1
        i = i0
        n = self.n
        done = False

        while not done:
            j = j0
            while True:
                if (self.C[i, j] == 0) and (not self.row_covered[i]) and (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j = (j + 1) % n
                if j == j0:
                    break
            i = (i + 1) % n
            if i == i0:
                done = True

        return (row, col)

    def __find_star_in_row(self, row: int) -> int:
        """Return the column index of the first starred element in the row.

        Returns -1 if that row has no starred element.
        """
        cols = np.where(self.marked[row] == STARRED)[0]
        return cols[0] if len(cols) > 0 else -1

    def __find_star_in_col(self, col: int) -> int:
        """Return the row index of the first starred element in the column.

        Returns -1 if that column has no starred element.
        """
        rows = np.where(self.marked[:, col] == STARRED)[0]
        return rows[0] if len(rows) > 0 else -1

    def __find_prime_in_row(self, row: int) -> int:
        """Return the column index of the first primed element in the row.

        Returns -1 if that row has no primed element.
        """
        cols = np.where(self.marked[row] == PRIMED)[0]
        return cols[0] if len(cols) > 0 else -1

    def __convert_path(self, path: IntMatrix, count: int) -> None:
        for i in range(count + 1):
            if self.marked[path[i, 0], path[i, 1]] == STARRED:
                self.marked[path[i, 0], path[i, 1]] = 0
            else:
                self.marked[path[i, 0], path[i, 1]] = STARRED

    def __reset_covers(self) -> None:
        """Clear all covered matrix cells."""
        self.row_covered = np.zeros(self.n, dtype=bool)
        self.col_covered = np.zeros(self.n, dtype=bool)

    def __erase_primes(self) -> None:
        """Erase all prime markings."""
        self.marked[self.marked == PRIMED] = 0


def pad_matrix(matrix: Matrix, pad_value: int = 0) -> Matrix:
    """Pad a possibly non-square matrix to make it square.

    Args:
        matrix: The matrix to pad.
        pad_value: The value to use for padding. Defaults to 0.

    Returns:
        The padded square matrix.
    """
    rows, cols = matrix.shape
    size = max(rows, cols)

    # Pad to square matrix
    if rows == size and cols == size:
        return matrix

    pad_width = ((0, size - rows), (0, size - cols))
    return np.pad(matrix, pad_width, mode="constant", constant_values=pad_value)


def make_cost_matrix[T: int | float](
    profit_matrix: ArrayLike, inversion_function: Callable[[T], T] | None = None
) -> Matrix:
    """Create a cost matrix from a profit matrix.

    Calls `inversion_function` to invert each profit value (defaults to
    ``max(matrix) - value``) and returns the resulting cost matrix.

    This is a static method. Call it like this:

        from munkres import Munkres
        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)

    For example:

        from munkres import Munkres
        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxsize - x)

    Args:
        profit_matrix: The matrix to convert from profit to cost values.
        inversion_function: The function to use to invert each entry in the profit matrix.

    Returns:
        A new matrix representing the inversion of `profix_matrix`.
    """
    arr = np.asarray(profit_matrix)
    if arr.ndim != 2:  # noqa: PLR2004
        raise ValueError("Input matrix must be 2D.")

    if inversion_function is None:
        maximum = np.max(arr)
        return maximum - arr  # type: ignore[no-any-return]

    return np.asarray([[inversion_function(value) for value in row] for row in arr])
