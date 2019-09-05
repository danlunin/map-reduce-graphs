from .lib import Graph, operations
from json import loads
import datetime
import calendar
import math


def divide(arg1, arg2):
    """Implements division"""
    return float(arg1) / arg2


def word_count_graph(input_stream: str, text_column: str, count_column: str,
                     from_file=False) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    graph0 = Graph().read_from_file(input_stream, loads) \
        if from_file else \
        Graph().read_from_iter(input_stream)
    return graph0\
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream: str, doc_column: str, text_column: str,
                         result_column: str,
                         from_file=False) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    graph0 = Graph().read_from_file(input_stream, loads) \
        if from_file \
        else Graph().read_from_iter(input_stream)
    """
    graph = graph0\
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    # Caution! Non-linear execution flow
    graph1 = graph.reduce(operations.Count('count'), [doc_column])
    graph1 = graph1.reduce(operations.Count('count'), [doc_column])
    graph2 = graph.sort(text_column)

    #print(graph.transactions)
    #print(graph1.transactions)
    #print(graph2.transactions)
    """
    def divide(row):
        return float(row[current_word_in_doc_count]) / row[total_words_in_doc]
    rows_count = 'rows_count'
    total_words_in_doc = 'total_in_doc'
    current_word_in_doc_count = 'word_in_doc'
    number_of_doc_with_word = 'docs_with_word_count'

    graph1 = graph0 \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    graph2 = graph0 \
        .count(operations.RowsCounter(rows_count), [doc_column])

    graph3 = graph1\
        .sort([doc_column, text_column])\
        .reduce(operations.FirstReducer(), keys=[doc_column, text_column])\
        .sort([doc_column]) \
        .join(operations.InnerJoiner(), graph2, keys=[doc_column]) \
        .sort([text_column])\
        .reduce(operations.Count(number_of_doc_with_word),
                [text_column, rows_count]) \
        .map(operations.Idf(number_of_doc_with_word, rows_count, 'idf')) \
        .sort([text_column])

    graph4 = graph1\
        .reduce(operations.Count(total_words_in_doc), [doc_column])\
        .sort([doc_column])

    def divide(row):
        return float(row[current_word_in_doc_count]) \
               / row[total_words_in_doc]
    graph5 = graph1 \
        .sort([doc_column, text_column]) \
        .reduce(operations.Count(current_word_in_doc_count),
                [doc_column, text_column]) \
        .sort([doc_column])\
        .join(operations.InnerJoiner(), graph4, keys=[doc_column])\
        .map(operations.ApplyFunction(divide, 'tf'))\
        .sort([text_column])

    graph6 = graph5.join(operations.InnerJoiner(), graph3, keys=[text_column])\
        .map(operations.Product(['tf', 'idf'], result_column)) \
        .sort([text_column, doc_column])\
        .map(operations.Project([doc_column, text_column, result_column])) \
        .reduce(operations.TopN(column=result_column, n=3),
                keys=[text_column]) \
        .sort([doc_column, text_column])

    return graph6


def pmi_graph(input_stream: str, doc_column: str, text_column: str,
              result_column: str, from_file=False) -> Graph:
    """
    Constructs graph which gives for every document the top 10 words
    ranked by pointwise mutual information
    """
    mentions = 'mentions'

    def filter_length(row):
        return len(row[text_column]) > 4

    def filter_number(row):
        return row['words_in_doc'] >= 2

    graph0 = Graph().read_from_file(input_stream, loads) \
        if from_file \
        else Graph().read_from_iter(input_stream)
    graph0 = graph0 \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .map(operations.Filter(condition=filter_length))\
        .sort([text_column, doc_column])

    # Count number of current word in each doc and filter words
    graph1 = graph0\
        .reduce(operations.Count('words_in_doc'),
                keys=[text_column, doc_column]) \
        .map(operations.Filter(condition=filter_number))\
        .sort([doc_column, text_column])

    # Join to get only needed words
    graph01 = graph1\
        .join(operations.InnerJoiner(), graph0, keys=[doc_column, text_column])

    # Count number of words in doc
    graph2 = graph01\
        .sort([doc_column])\
        .reduce(operations.Count('total_in_doc'), keys=[doc_column])\
        .sort([doc_column])

    # Count frequency in one doc
    graph22 = graph01\
        .join(operations.InnerJoiner(), graph2, keys=[doc_column])\
        .map(operations.ApplyFunction(lambda row: float(row['words_in_doc'])
                                      / row['total_in_doc'],
                                      'local_frequency'))

    graph4 = graph01\
        .sort([text_column, doc_column]) \
        .sort([text_column]) \
        .reduce(operations.Count(mentions), [text_column]) \
        .sort([text_column])

    # Prepare and leave final graph
    graph5 = graph4\
        .join(operations.InnerJoiner(), graph22, keys=[text_column])\
        .sort([doc_column])\
        .reduce(operations.FirstReducer(), keys=[doc_column, text_column])\
        .sort([doc_column])

    # Total words
    graph6 = graph01\
        .count(operations.RowsCounter('total_words'),
               [text_column, doc_column])\
        .sort([text_column])\
        .reduce(operations.FirstReducer(), keys=[text_column]) \
        .map(operations.Project([text_column, 'total_words']))

    graph7 = graph5.join(operations.InnerJoiner(), graph6, keys=[text_column])\
        .map(operations.ApplyFunction(lambda row: float(row[mentions])
                                      / row['total_words'],
                                      'global_frequency'))\
        .map(operations.Idf('global_frequency', 'local_frequency',
                            result_column))\
        .map(operations.Project([text_column, result_column, 'doc_id']))\
        .sort([doc_column])\
        .reduce(operations.TopN(column=result_column, n=10), keys=[doc_column])

    return graph7


def yandex_maps_graph(input_stream_time: str, input_stream_length: str,
                      enter_time_column: str, leave_time_column: str,
                      edge_id_column: str, start_coord_column: str,
                      end_coord_column: str,
                      weekday_result_column: str, hour_result_column: str,
                      speed_result_column: str,
                      from_file=False) -> Graph:
    """
    Constructs graph which measures average speed in km/h depending
     on the weekday and hour
    """

    def distance(row):
        """Computes distance between objects based on lon and lat"""
        radius = 6371.0  # km
        lon1, lat1 = row[start_coord_column]
        lon2, lat2 = row[end_coord_column]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = radius * c
        return d

    def get_weekday(row):
        """
        Returns first three letters from weekday's name extracted from date
        """
        try:
            day_number = datetime.datetime.\
                strptime(row[enter_time_column], '%Y%m%dT%H%M%S.%f').weekday()
        except ValueError:
            day_number = datetime.datetime. \
                strptime(row[enter_time_column], '%Y%m%dT%H%M%S').weekday()
        return calendar.day_name[day_number][:3]

    def parse_date(arg, with_milliseconds=True):
        date_format = '%Y%m%dT%H%M%S.%f' if with_milliseconds \
            else '%Y%m%dT%H%M%S'
        return datetime.datetime.strptime(arg, date_format)

    def get_hour(row):
        """Returns hour extracted from given date"""
        arg1 = row[enter_time_column]
        try:
            return parse_date(arg1).hour
        except ValueError:
            return parse_date(arg1, with_milliseconds=False).hour

    def get_diff_in_hours(row):
        arg1 = row[enter_time_column]
        arg2 = row[leave_time_column]
        try:
            start_date = parse_date(arg1)
        except ValueError:
            start_date = parse_date(arg1, with_milliseconds=False)
        try:
            end_date = parse_date(arg2)
        except ValueError:
            end_date = parse_date(arg2, with_milliseconds=False)
        return (end_date - start_date).total_seconds()/3600

    graph0 = Graph().read_from_file(input_stream_length, loads) \
        if from_file \
        else Graph().read_from_iter(input_stream_length)

    graph0 = graph0 \
        .map(operations.ApplyFunction(distance, 'distance'))

    graph1 = Graph().read_from_file(input_stream_time, loads) \
        if from_file \
        else Graph().read_from_iter(input_stream_time)

    graph1 = graph1 \
        .map(operations.ApplyFunction(get_diff_in_hours, 'hours'))\
        .sort([edge_id_column])\
        .join(operations.InnerJoiner(), graph0, keys=[edge_id_column])\
        .map(operations.ApplyFunction(lambda row: float(row['distance'])
                                      / row['hours'],
                                      speed_result_column))\
        .map(operations.ApplyFunction(get_weekday, weekday_result_column))\
        .map(operations.ApplyFunction(get_hour, hour_result_column))\
        .sort([weekday_result_column, hour_result_column]) \
        .map(operations.Project([weekday_result_column,
                                 hour_result_column, 'speed']))\
        .reduce(operations.Average(column='speed'),
                keys=[weekday_result_column, hour_result_column])

    return graph1
