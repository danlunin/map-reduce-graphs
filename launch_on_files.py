import graphs
from json import dump
import os


def get_absolute_input_path(filename):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'resource', filename)


# tf-idf
graph = graphs.inverted_index_graph('file', doc_column='doc_id',
                                    text_column='text',
                                    result_column='tf_idf', from_file=True)
result = graph.run(file=get_absolute_input_path('text_corpus.txt'))

with open("tf_idf.txt", 'w') as j:
    dump(result, j)


# word count
graph = graphs.word_count_graph('file', text_column='text',
                                count_column='count',
                                from_file=True)
result = graph.run(file=get_absolute_input_path('text_corpus.txt'))
with open("word_count.txt", 'w') as j:
    dump(result, j)

# Yandex maps
graph = graphs.yandex_maps_graph(
        'travel_time', 'edge_length',
        enter_time_column='enter_time', leave_time_column='leave_time',
        edge_id_column='edge_id',
        start_coord_column='start', end_coord_column='end',
        weekday_result_column='weekday', hour_result_column='hour',
        speed_result_column='speed',
        from_file=True
    )


result = graph.run(
        travel_time=get_absolute_input_path('travel_times.txt'),
        edge_length='resource/road_graph_data.txt'
    )

with open("yandex_maps.txt", 'w') as j:
    dump(result, j)

# pmi

graph = graphs.pmi_graph('file', doc_column='doc_id', text_column='text',
                         result_column='pmi',
                         from_file=True)
result = graph.run(file=get_absolute_input_path('text_corpus.txt'))
with open("pmi.txt", 'w') as j:
    dump(result, j)
