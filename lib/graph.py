from typing import Sequence, Callable, List
import uuid
from itertools import tee

from .operations import Row, Mapper, Reducer, Joiner, Map, \
    Reduce, Sort, Join, CountAll, Read, ReadFromFile


class Graph:
    """Computational graph implementation"""
    def __init__(self, parents=None, data_source=None, parser=None,
                 operation=None):
        self.__data_source = data_source
        self.__parents = parents if parents is not None else []
        self.__parser = parser
        self.__operation = operation
        self.__id = str(uuid.uuid1())

    def read_from_iter(self, name: str) -> 'Graph':
        """
        Construct new graph extended with operation which adds data
        (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        :param name: name of kwarg to use as data source
        """
        self.__data_source = name
        return Graph(data_source=self.__data_source,
                     parents=[self], parser=self.__parser)

    def read_from_file(self, filename: str,
                       parser: Callable[[str], Row]) -> 'Graph':
        """Construct new graph extended with operation
        for reading rows from file
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        self.__data_source = filename
        self.__parser = parser
        return Graph(data_source=self.__data_source,
                     parents=[self], parser=self.__parser)

    def map(self, mapper: Mapper) -> 'Graph':
        """Construct new graph extended with map operation
        with particular mapper
        :param mapper: mapper to use
        """
        return Graph(data_source=self.__data_source,
                     parents=[self], parser=self.__parser,
                     operation=Map(mapper))

    def reduce(self, reducer: Reducer, keys: Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation
        with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        return Graph(data_source=self.__data_source,
                     parents=[self], parser=self.__parser,
                     operation=Reduce(reducer, keys=keys))

    def count(self, counter: Reducer, keys: Sequence[str]) -> 'Graph':
        """Construct new graph extended with count operation
        with particular reducer
        :param counter: reducer to use
        :param keys: keys for grouping
        """
        return Graph(data_source=self.__data_source,
                     parents=[self], parser=self.__parser,
                     operation=CountAll(counter, keys=keys))

    def sort(self, keys: Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        return Graph(data_source=self.__data_source,
                     parents=[self], parser=self.__parser,
                     operation=Sort(keys=keys))

    def join(self, joiner: Joiner, join_graph: 'Graph',
             keys: Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        return Graph(data_source=self.__data_source,
                     parents=[self, join_graph], parser=self.__parser,
                     operation=Join(joiner, keys=keys))

    def run_recursively(self, kwargs, cache):
        if len(self.__parents) == 2:
            parent1 = self.__parents[0].run_recursively(kwargs, cache)
            parent2 = self.__parents[1].run_recursively(kwargs, cache)
            return self.__operation(parent1, parent2)

        elif self.__operation is not None:
            if self.__id in cache:
                print("It's a cache")
                current, copy = tee(cache[self.__id])
                return copy
            else:
                current, calculate = \
                    tee(self.__operation(
                        self.__parents[0].run_recursively(kwargs, cache)))
                cache[self.__id] = calculate
                return current
        else:
            if self.__parser is not None:
                return ReadFromFile(self.__parser)(kwargs[self.__data_source])
            return Read()(kwargs[self.__data_source])

    def run(self, **kwargs) -> List[Row]:
        """Single method to start execution; data sources passed as kwargs"""
        cache = {}
        return list(self.run_recursively(kwargs, cache))
