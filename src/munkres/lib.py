"""Munkres (Hungarian) algorithm implementation.

The Munkres module provides an implementation of the Munkres algorithm
(also called the Hungarian algorithm or the Kuhn-Munkres algorithm), useful for
solving the Assignment Problem.

For complete usage documentation, see https://software.clapper.org/munkres/.
"""

import sys
from collections.abc import Callable, Sequence

type AnyNum = int | float
type Matrix = list[list[AnyNum]]
type MatrixLike = Sequence[Sequence[AnyNum]]
type IntMatrix = list[list[int]]


# Constants
class _Disallowed:
    """Sentinel identifying disallowed assignments."""


DISALLOWED = _Disallowed()
DISALLOWED_PRINTVAL = "D"
STARRED = 1
PRIMED = 2


class UnsolvableMatrixError(Exception):
    """Exception raised for unsolvable matrices."""


class Munkres:
    """Compute the Munkres solution to the classical assignment problem.

    See the module documentation for usage instructions and background.
    """

    def __init__(self) -> None:
        """Initialize the solver state."""
        self.C: Matrix = []
        self.row_covered: list[bool] = []
        self.col_covered: list[bool] = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked: IntMatrix = []
        self.path: IntMatrix = []
        self.original_length = 0
        self.original_width = 0

    def pad_matrix(self, matrix: MatrixLike, pad_value: int = 0) -> Matrix:
        """Pad a possibly non-square matrix to make it square.

        **Parameters**

        - `matrix` (list of lists of numbers): matrix to pad
        - `pad_value` (`int`): value to use to pad the matrix

        **Returns**

        a new, possibly padded, matrix
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix: Matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = list(row)
            if total_rows > row_len:
                # Row too short. Pad it.
                new_row += [pad_value] * (total_rows - row_len)
            new_matrix.append(new_row)

        while len(new_matrix) < total_rows:
            new_matrix.append([pad_value] * total_rows)

        return new_matrix

    def compute(self, cost_matrix: MatrixLike) -> Sequence[tuple[int, int]]:
        """Return the indexes for the lowest-cost assignments.

        Returns a list of `(row, column)` tuples that can be used to traverse the
        matrix.

        **WARNING**: This code handles square and rectangular matrices. It
        does *not* handle irregular matrices.

        **Parameters**

        - `cost_matrix` (list of lists of numbers): The cost matrix. If this
          cost matrix is not square, it will be padded with zeros, via a call
          to `pad_matrix()`. (This method does *not* modify the caller's
          matrix. It operates on a copy of the matrix.)


        **Returns**

        A list of `(row, column)` tuples that describe the lowest cost path
        through the matrix.
        """
        self.C = self.pad_matrix(cost_matrix)
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0])
        self.row_covered = [False for _ in range(self.n)]
        self.col_covered = [False for _ in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_int_matrix(self.n * 2, 0)
        self.marked = self.__make_int_matrix(self.n, 0)

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
        return [
            (i, j)
            for i in range(self.original_length)
            for j in range(self.original_width)
            if self.marked[i][j] == STARRED
        ]

    def __make_int_matrix(self, n: int, val: int) -> IntMatrix:
        """Create an n x n matrix populated with the provided value."""
        matrix: IntMatrix = []
        for _ in range(n):
            matrix.append([val for _ in range(n)])
        return matrix

    def __step1(self) -> int:
        """Normalize rows by subtracting their minimum element."""
        n = self.n
        for i in range(n):
            vals = [x for x in self.C[i] if x is not DISALLOWED]
            if len(vals) == 0:
                # All values in this row are DISALLOWED. This matrix is unsolvable.
                raise UnsolvableMatrixError(f"Row {i} is entirely DISALLOWED.")
            minval = min(vals)
            # Find the minimum value for this row and subtract that minimum from every element.
            for j in range(n):
                if self.C[i][j] is not DISALLOWED:
                    self.C[i][j] -= minval
        return 2

    def __step2(self) -> int:
        """Star uncovered zeros that have no starred zero in their row or column."""
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.C[i][j] == 0) and (not self.col_covered[j]) and (not self.row_covered[i]):
                    self.marked[i][j] = STARRED
                    self.col_covered[j] = True
                    self.row_covered[i] = True
                    break

        self.__clear_covers()
        return 3

    def __step3(self) -> int:
        """Cover each column that contains a starred zero.

        If K columns are covered, the starred zeros describe a complete set of unique
        assignments. In this case, go to DONE; otherwise go to Step 4.
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == STARRED and not self.col_covered[j]:
                    self.col_covered[j] = True
                    count += 1

        return 7 if count >= n else 4

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
                self.marked[row][col] = PRIMED
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
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count - 1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count - 1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self) -> int:
        """Adjust uncovered columns and covered rows by the smallest uncovered value.

        Return to Step 4 without altering any stars, primes, or covered lines.
        """
        minval = self.__find_smallest()
        events = 0  # track actual changes to matrix
        for i in range(self.n):
            for j in range(self.n):
                if self.C[i][j] is DISALLOWED:
                    continue
                if self.row_covered[i]:
                    self.C[i][j] += minval
                    events += 1
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
                    events += 1
                if self.row_covered[i] and not self.col_covered[j]:
                    events -= 2  # change reversed, no real difference
        if events == 0:
            raise UnsolvableMatrixError("Matrix cannot be solved!")
        return 4

    def __find_smallest(self) -> AnyNum:
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxsize
        for i in range(self.n):
            for j in range(self.n):
                if (
                    (not self.row_covered[i])
                    and (not self.col_covered[j])
                    and self.C[i][j] is not DISALLOWED
                    and minval > self.C[i][j]
                ):
                    minval = self.C[i][j]
        return minval

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
                if (self.C[i][j] == 0) and (not self.row_covered[i]) and (not self.col_covered[j]):
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
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == STARRED:
                col = j
                break

        return col

    def __find_star_in_col(self, col: int) -> int:
        """Return the row index of the first starred element in the column.

        Returns -1 if that column has no starred element.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == STARRED:
                row = i
                break

        return row

    def __find_prime_in_row(self, row: int) -> int:
        """Return the column index of the first primed element in the row.

        Returns -1 if that row has no primed element.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == PRIMED:
                col = j
                break

        return col

    def __convert_path(self, path: Sequence[Sequence[int]], count: int) -> None:
        for i in range(count + 1):
            if self.marked[path[i][0]][path[i][1]] == STARRED:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = STARRED

    def __clear_covers(self) -> None:
        """Clear all covered matrix cells."""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self) -> None:
        """Erase all prime markings."""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == PRIMED:
                    self.marked[i][j] = 0


def make_cost_matrix(
    profit_matrix: MatrixLike, inversion_function: Callable[[AnyNum], AnyNum] | None = None
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

    **Parameters**

    - `profit_matrix` (list of lists of numbers): The matrix to convert from
       profit to cost values.
    - `inversion_function` (`function`): The function to use to invert each
       entry in the profit matrix.

    **Returns**

    A new matrix representing the inversion of `profix_matrix`.
    """
    if inversion_function is None:
        maximum = max(max(row) for row in profit_matrix)

        def default_inversion(value: AnyNum, /, max_value: AnyNum = maximum) -> AnyNum:
            return max_value - value

        inversion_function = default_inversion

    return [[inversion_function(value) for value in row] for row in profit_matrix]


def print_matrix(matrix: Matrix, msg: str | None = None) -> None:
    """Convenience function: Displays the contents of a matrix.

    **Parameters**

    - `matrix` (list of lists of numbers): The matrix to print
    - `msg` (`str`): Optional message to print before displaying the matrix
    """
    if msg is not None:
        sys.stdout.write(f"{msg}\n")

    # Calculate the appropriate format width.
    width = 0
    for row in matrix:
        for value in row:
            display_value = DISALLOWED_PRINTVAL if value is DISALLOWED else value
            width = max(width, len(str(display_value)))

    fmt = f"{{:>{width}}}"

    for row in matrix:
        sep = "["
        for value in row:
            display_value = DISALLOWED_PRINTVAL if value is DISALLOWED else value
            formatted = fmt.format(display_value)
            sys.stdout.write(f"{sep}{formatted}")
            sep = ", "
        sys.stdout.write("]\n")


if __name__ == "__main__":
    matrices = [
        # Square
        ([[400, 150, 400], [400, 450, 600], [300, 225, 300]], 850),  # expected cost
        # Rectangular variant
        ([[400, 150, 400, 1], [400, 450, 600, 2], [300, 225, 300, 3]], 452),  # expected cost
        # Square
        ([[10, 10, 8], [9, 8, 1], [9, 7, 4]], 18),
        # Square variant with floating point value
        ([[10.1, 10.2, 8.3], [9.4, 8.5, 1.6], [9.7, 7.8, 4.9]], 19.5),
        # Rectangular variant
        ([[10, 10, 8, 11], [9, 8, 1, 1], [9, 7, 4, 10]], 15),
        # Rectangular variant with floating point value
        ([[10.01, 10.02, 8.03, 11.04], [9.05, 8.06, 1.07, 1.08], [9.09, 7.1, 4.11, 10.12]], 15.2),
        # Rectangular with DISALLOWED
        (
            [
                [4, 5, 6, DISALLOWED],
                [1, 9, 12, 11],
                [DISALLOWED, 5, 4, DISALLOWED],
                [12, 12, 12, 10],
            ],
            20,
        ),
        # Rectangular variant with DISALLOWED and floating point value
        (
            [
                [4.001, 5.002, 6.003, DISALLOWED],
                [1.004, 9.005, 12.006, 11.007],
                [DISALLOWED, 5.008, 4.009, DISALLOWED],
                [12.01, 12.011, 12.012, 10.013],
            ],
            20.028,
        ),
        # DISALLOWED to force pairings
        (
            [
                [1, DISALLOWED, DISALLOWED, DISALLOWED],
                [DISALLOWED, 2, DISALLOWED, DISALLOWED],
                [DISALLOWED, DISALLOWED, 3, DISALLOWED],
                [DISALLOWED, DISALLOWED, DISALLOWED, 4],
            ],
            10,
        ),
        # DISALLOWED to force pairings with floating point value
        (
            [
                [1.1, DISALLOWED, DISALLOWED, DISALLOWED],
                [DISALLOWED, 2.2, DISALLOWED, DISALLOWED],
                [DISALLOWED, DISALLOWED, 3.3, DISALLOWED],
                [DISALLOWED, DISALLOWED, DISALLOWED, 4.4],
            ],
            11.0,
        ),
    ]

    m = Munkres()
    for cost_matrix, expected_total in matrices:
        print_matrix(cost_matrix, msg="cost matrix")
        indexes = m.compute(cost_matrix)
        total_cost = 0
        for r, c in indexes:
            x = cost_matrix[r][c]
            total_cost += x
            sys.stdout.write(f"({r}, {c}) -> {x}\n")
        sys.stdout.write(f"lowest cost={total_cost}\n")
        assert expected_total == total_cost
