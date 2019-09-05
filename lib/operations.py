from abc import abstractmethod, ABC
from types import FunctionType
from typing import NewType, Dict, Any, Generator, Iterable, Tuple, Sequence
import string
from itertools import groupby, chain, tee
import math

Row = NewType('Row', Dict[str, Any])
OperationResult = NewType('OperationResult', Generator[Row, None, None])


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: Iterable[Row], *args) -> OperationResult:
        pass


# Operations


class Mapper(ABC):
    """Base class for mappers"""
    @abstractmethod
    def __call__(self, row: Row) -> OperationResult:
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper):
        self.mapper = mapper

    def __call__(self, rows: Iterable[Row], *args) -> OperationResult:
        for row in rows:
            yield from self.mapper(row)


class Reducer(ABC):
    """Base class for reducers"""
    @abstractmethod
    def __call__(self, group_key: Tuple[str], rows: Iterable[Row]) \
            -> OperationResult:
        pass


class Reduce(Operation):
    def _leave_only_keys(self, row):
        return [row[k] for k in self.keys]

    def __init__(self, reducer: Reducer, keys: Sequence[str]):
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: Iterable[Row], *args) -> OperationResult:
        for key, group in groupby(rows, self._leave_only_keys):
            yield from self.reducer(tuple(self.keys), list(group))


class CountAll(Operation):
    def __init__(self, counter: Reducer, keys: Sequence[str]):
        self.reducer = counter
        self.keys = keys

    def __call__(self, rows: Iterable[Row], *args) -> OperationResult:
        yield from self.reducer(tuple(self.keys), list(rows))


class Sort(Operation):
    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def __call__(self, rows: Iterable[Row], *args) -> OperationResult:
        yield from sorted(rows,
                          key=lambda row: [row[k] for k in self.keys])


class Joiner(ABC):
    """Base class for joiners"""
    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2'):
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @staticmethod
    def next(iterator: Iterable[Tuple]) -> Tuple[Dict, Iterable[Dict]]:
        try:
            return next(iterator)
        except StopIteration:
            return (None, None)

    @abstractmethod
    def __call__(self,
                 keys: Sequence[str],
                 rows_a: Iterable[Row],
                 rows_b: Iterable[Row]) -> OperationResult:
        pass

    def _merge_row(self, keys: Sequence[str],
                   row_left: Row, row_right: Row) -> Dict:
        new_row = {}
        duplicate_keys = [key for key in row_left.keys()
                          if key in row_right.keys() and key not in keys]
        left_keys = [key for key in row_left.keys()
                     if key not in row_right.keys()]
        right_keys = [key for key in row_right.keys()
                      if key not in row_left.keys()]

        for key in duplicate_keys:
            new_row[key + self._a_suffix] = row_left[key]
            new_row[key + self._b_suffix] = row_right[key]

        for key in left_keys:
            new_row[key] = row_left[key]

        for key in right_keys:
            new_row[key] = row_right[key]

        for key in keys:
            new_row[key] = row_left[key]

        yield new_row

    def _simple_join(self, rows_a, rows_b, keys):
        a_rows_list = list(rows_a)
        b_rows_list = list(rows_b)
        for row_a in a_rows_list:
            for row_b in b_rows_list:
                yield from self._merge_row(keys, row_a, row_b)


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: Iterable[Row], *args) -> OperationResult:

        def check_move_right(left_key, right_key):
            return (left_key is not None and right_key is not None)\
                   and right_key < left_key or left_key is None

        def check_move_left(left_key, right_key):
            return left_key is not None and \
                    right_key is not None and \
                    left_key < right_key \
                    or right_key is None

        left_pointer = groupby(Sort(self.keys)(rows),
                               lambda row: tuple(row[key]
                                                 for key in self.keys))
        right_pointer = groupby(Sort(self.keys)(args[0]),
                                lambda row: tuple(row[key]
                                                  for key in self.keys))

        left_key, left_rows_group = Joiner.next(left_pointer)
        right_key, right_rows_group = Joiner.next(right_pointer)

        while left_key is not None or right_key is not None:
            if check_move_right(left_key, right_key):
                yield from self.joiner(self.keys, None, right_rows_group)
                right_key, right_rows_group = Joiner.next(right_pointer)

            elif check_move_left(left_key, right_key):
                yield from self.joiner(self.keys, left_rows_group, None)
                left_key, left_rows_group = Joiner.next(left_pointer)
            else:
                yield from self.joiner(self.keys, left_rows_group,
                                       right_rows_group)
                left_key, left_rows_group = Joiner.next(left_pointer)
                right_key, right_rows_group = Joiner.next(right_pointer)


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""
    def __call__(self, row: Row) -> OperationResult:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""
    def __call__(self, group_key: Tuple[str],
                 rows: Iterable[Row]) -> OperationResult:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: Row) -> OperationResult:
        punctuation = set(string.punctuation)
        row[self.column] = ''.join(char for char in
                                   row[self.column] if char not in punctuation)
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: Row) -> OperationResult:
        row[self.column] = row[self.column].lower()
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""
    def __init__(self, column: str, separator: str = None):
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: Row) -> OperationResult:
        splitted = row[self.column].split(self.separator)
        for column in splitted:
            new_raw = row.copy()
            new_raw[self.column] = column
            yield new_raw


class Product(Mapper):
    """Calculates product of multiple columns"""
    def __init__(self, columns: Sequence[str],
                 result_column: str = 'product'):
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: Row) -> OperationResult:
        row[self.result_column] = 1
        for key in self.columns:
            row[self.result_column] *= row[key]
        yield row


class Idf(Mapper):
    """Calculates idf"""
    def __init__(self, column1: str, column2: str,
                 result_column: str = 'product'):
        """
        :param column1: enumerator
        :param column2: denominator
        :param result_column: column for log of division
        """
        self.column1 = column1
        self.column2 = column2
        self.result_column = result_column

    def __call__(self, row: Row) -> OperationResult:
        row[self.result_column] = \
            math.log(row[self.column2] / row[self.column1])
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""
    def __init__(self, condition: FunctionType):
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: Row) -> OperationResult:
        if self.condition(row):
            yield row


class ApplyFunction(Mapper):
    """Applies function to two given columns"""
    def __init__(self, func: FunctionType, result_column: str = 'result'):
        """
        :param column1: first argument
        :param column2: second argument
        :param result_column: column name to save function result in
        """
        self.result_column = result_column
        self.func = func

    def __call__(self, row: Row) -> OperationResult:
        row[self.result_column] = self.func(row)
        yield row


class Read:
    """Reads from iterator"""
    def __init__(self):
        pass

    def __call__(self, data) -> OperationResult:
        yield from data


class ReadFromFile:
    """Reads from file using given parser"""
    def __init__(self, parse_function):
        self.parse_function = parse_function

    def __call__(self, file_name) -> OperationResult:
        with open(file_name) as f:
            for line in f:
                yield self.parse_function(line)


class Project(Mapper):
    """Leave only mentioned columns"""
    def __init__(self, columns: Sequence[str]):
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: Row) -> OperationResult:
        new_row = {}
        for column in self.columns:
            new_row[column] = row[column]
        yield new_row


# Reducers


class TopN(Reducer):
    """Return top N by value"""
    def __init__(self, column: str, n: int):
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: Tuple[str],
                 rows: Iterable[Row])-> OperationResult:
        sorted_rows = sorted(rows,
                             key=lambda row: row[self.column_max],
                             reverse=True)
        sorted_rows = sorted_rows[:self.n]
        yield from sorted_rows


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""
    def __init__(self, words_column: str, result_column: str = 'tf'):
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: Tuple[str],
                 rows: Iterable[Row]) -> OperationResult:
        rows = sorted(list(rows), key=lambda row: row[self.words_column])
        for key, group in groupby(rows, lambda row: row[self.words_column]):
            group, group_copy = tee(group)
            group_size = 0
            if group_copy is not None:
                for row_sample in group_copy:
                    group_size += 1
                needed_columns = set(group_key)
                needed_columns.add(self.words_column)
                new_row = {key: row_sample[key] for key in row_sample.keys()
                           if key in needed_columns}
                new_row[self.result_column] = \
                    float(group_size) / float(len(rows))
                yield new_row


class Count(Reducer):
    """Count rows passed and yield single row as a result"""
    def __init__(self, column: str):
        """
        :param column: name of column to count
        """
        self.column = column

    def __call__(self, group_key: Tuple[str],
                 rows: Iterable[Row]) -> OperationResult:
        rows, rows_copy = tee(rows)
        rows_size = 0
        for _ in rows_copy:
            rows_size += 1
        for row in rows:
            new_row = {key: row[key] for key in group_key}
            new_row[self.column] = rows_size
        yield new_row


class RowsCounter(Reducer):
    """Count rows and save number to result column"""
    def __init__(self, column: str):
        """
        :param column: name of column to count
        """
        self.column = column

    def __call__(self, group_key: Tuple[str],
                 rows: Iterable[Row]) -> OperationResult:
        rows, rows_copy = tee(rows)
        rows_size = 0
        for _ in rows_copy:
            rows_size += 1
        for row in rows:
            new_row = {key: row[key] for key in group_key}
            new_row[self.column] = rows_size
            yield new_row


class Sum(Reducer):
    """Sum values in column passed and yield single row as a result"""
    def __init__(self, column: str):
        """
        :param column: name of column to sum
        """
        self.column = column

    def __call__(self, group_key: Tuple[str],
                 rows: Iterable[Row]) -> OperationResult:
        total_sum = 0
        for row in rows:
            total_sum += row[self.column]
        for row in rows:
            new_row = {key: row[key] for key in group_key}
            new_row[self.column] = total_sum
            yield new_row
            break


class Average(Reducer):
    """Returns average for the group"""

    def __init__(self, column: str):
        """
        :param column: name of column to sum
        """
        self.column = column

    def __call__(self, group_key: Tuple[str],
                 rows: Iterable[Row]) -> OperationResult:
        total_sum = 0
        rows, rows_copy = tee(rows)
        rows_size = 0
        for row in rows_copy:
            total_sum += row[self.column]
            rows_size += 1
        for row in rows:
            new_row = {key: row[key] for key in group_key}
            new_row[self.column] = float(total_sum / rows_size)
            yield new_row
            break

# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""
    def __call__(self,
                 keys: Sequence[str],
                 rows_a: Iterable[Row],
                 rows_b: Iterable[Row]) -> OperationResult:
        if rows_a is not None and rows_b is not None:
            yield from self._simple_join(rows_a, rows_b, keys)


class OuterJoiner(Joiner):
    """Join with outer strategy"""
    def __call__(self,
                 keys: Sequence[str],
                 rows_a: Iterable[Row],
                 rows_b: Iterable[Row]) -> OperationResult:
        if rows_a is not None and rows_b is not None:
            yield from self._simple_join(rows_a, rows_b, keys)
        elif rows_a is not None:
            yield from rows_a
        else:
            yield from rows_b


class LeftJoiner(Joiner):
    """Join with left strategy"""
    def __call__(self, keys: Sequence[str],
                 rows_a: Iterable[Row],
                 rows_b: Iterable[Row]) -> OperationResult:
        if rows_a is not None and rows_b is not None:
            yield from self._simple_join(rows_a, rows_b, keys)
        elif rows_a is not None:
            yield from rows_a


class RightJoiner(Joiner):
    """Join with right strategy"""
    def __call__(self, keys: Sequence[str],
                 rows_a: Iterable[Row],
                 rows_b: Iterable[Row]) -> OperationResult:
        if rows_a is not None and rows_b is not None:
            yield from self._simple_join(rows_a, rows_b, keys)
        elif rows_b is not None:
            yield from rows_b
