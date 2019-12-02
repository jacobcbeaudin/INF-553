from pyspark import SparkContext
from itertools import islice, combinations, tee
from operator import add
from sys import argv


def basket_mapper(element):
    comma_location = element.find(',')
    if int(argv[1]) == 1:
        return element[:comma_location], element[(comma_location+1):]
    elif int(argv[1]) == 2:
        return element[(comma_location + 1):], element[:comma_location]
    else:
        print('Error: please enter a valid case #: {1,2}')

def read_baskets():
    # Spark Context intitialize
    sc = SparkContext('local[*]', 'hw2_task_1')
    sc.setLogLevel("OFF")
    baskets = sc.textFile(argv[3]).\
        mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element).\
        map(basket_mapper).groupByKey().\
        map(lambda basket: basket[1]).map(set) #.persist()
    return sc, baskets


def set_up_task1():
    # create basket depending on case from command prompt
    sc, baskets = read_baskets()

    # use to calculate sample support
    sample_support_vars = dict(num_baskets=baskets.count(), support=int(argv[2]))
    # sample_support_vars = dict(num_baskets=baskets.count(), support=7)

    return sc, baskets, sample_support_vars


def declare_son_vars():
    # count of pass number
    frequent_item_set_candidates = dict()
    frequent_item_sets = dict()
    return frequent_item_set_candidates, frequent_item_sets


def prune_baskets(basket, previous_frequent_item_sets):
    return basket.intersection(set(previous_frequent_item_sets))


def prune_candidates(n_baskets_partition, sample_support_vars, candidate_ids_dict):
    sample_support = sample_support_vars['support'] * n_baskets_partition / sample_support_vars['num_baskets']
    frequent_item_set_candidates = tuple(k for k, v in candidate_ids_dict.items() if v >= sample_support)
    return frequent_item_set_candidates


def update_item_set_dict(item_set_dict, item_set_candidate):
    if item_set_candidate in item_set_dict:
        item_set_dict[item_set_candidate] += 1
    else:
        item_set_dict[item_set_candidate] = 1
    return item_set_dict


def phase_one(iterator, sample_support_vars, previous_frequent_item_sets, item_set_size):
    possible_item_set_candidate_dict = dict()
    n_baskets_partition = 0
    if item_set_size > 2:
        iterator = list(iterator)
        for _ in iterator:
            n_baskets_partition += 1

        for i in range(previous_frequent_item_sets.__len__()-1):
            for j in range(i+1, previous_frequent_item_sets.__len__()):
                if previous_frequent_item_sets[i][:(item_set_size-2)] == \
                        previous_frequent_item_sets[j][:(item_set_size-2)]:
                    possible_item_set_candidate = tuple(sorted(set(
                        previous_frequent_item_sets[i]).union(set(previous_frequent_item_sets[j]))))
                    # BUG IS HERE!!!!!
                    for partitioned_basket in iterator:
                        if set(possible_item_set_candidate).issubset(partitioned_basket):
                            possible_item_set_candidate_dict = update_item_set_dict(possible_item_set_candidate_dict,
                                                                                    possible_item_set_candidate)

    elif item_set_size == 2:
        for partitioned_basket in iterator:
            n_baskets_partition += 1
            # go through all combinations of possible item sets by evaluating frequent singletons
            for possible_item_set_candidate in map(tuple, combinations(previous_frequent_item_sets, item_set_size)):
                if set(possible_item_set_candidate).issubset(partitioned_basket):
                    possible_item_set_candidate_dict = update_item_set_dict(possible_item_set_candidate_dict,
                                                                            possible_item_set_candidate)
    else:
        # enumerate each item in each basket
        for partitioned_basket in iterator:
            partitioned_basket = list(partitioned_basket)
            n_baskets_partition += 1
            for possible_item_set_candidate in partitioned_basket:
                possible_item_set_candidate_dict = update_item_set_dict(possible_item_set_candidate_dict,
                                                                        possible_item_set_candidate)

    frequent_item_set_candidates = prune_candidates(n_baskets_partition, sample_support_vars, possible_item_set_candidate_dict)

    yield frequent_item_set_candidates


def phase_two(iterator, frequent_item_set_candidates, item_set_length):
    frequent_item_set_dict = dict()
    if item_set_length > 1:
        for basket in iterator:
            for frequent_item_set_candidate in frequent_item_set_candidates:
                if set(frequent_item_set_candidate).issubset(basket):
                    frequent_item_set_dict = update_item_set_dict(frequent_item_set_dict, frequent_item_set_candidate)
        if item_set_length != 2:
            frequent_item_set = set((k, v) for k, v in frequent_item_set_dict.items())
        else:
            frequent_item_set = set((k, v) for k, v in frequent_item_set_dict.items())
    else:
        for basket in iterator:
            for frequent_item_set_candidate in basket:
                frequent_item_set_dict = update_item_set_dict(frequent_item_set_dict, frequent_item_set_candidate)
        frequent_item_set = set((k, v) for k, v in frequent_item_set_dict.items())

    yield frequent_item_set


def write_output_file(item_set_size, frequent_item_set_candidates, frequent_item_sets):
    f = open(argv[4], 'w')
    f.write('Candidates:\n')

    if int(argv[1]) == 1:
        frequent_item_sets[frequent_item_sets.__len__()] = frequent_item_sets[frequent_item_sets.__len__()][0]
        frequent_item_set_candidates[frequent_item_set_candidates.__len__()] = \
        frequent_item_set_candidates[frequent_item_set_candidates.__len__()][0]
        for i in range(1, item_set_size):
            if i == 2:
                f.write('\n')
            if i != 1:
                if i < item_set_size - 1:
                    f.write(str(frequent_item_set_candidates[i])[1:-1].replace('), ', '),') + '\n\n')
                else:
                    f.write(str(frequent_item_set_candidates[i]))  # [1:-2])
            else:
                singletons = frequent_item_set_candidates[i]
                for j, frequent_singleton in enumerate(singletons):
                    if j != singletons.__len__() - 1:
                        f.write('(\'' + str(frequent_singleton) + '\'),')
                    else:
                        f.write('(\'' + str(frequent_singleton) + '\')\n')

        f.write('\n\nFrequent Itemsets:\n')

        for i in range(1, item_set_size):
            if i == 2:
                f.write('\n')
            if i != 1:
                if i < item_set_size - 1:
                    f.write(str(frequent_item_sets[i])[1:-1].replace('), ', '),') + '\n\n')
                else:
                    f.write(str(frequent_item_sets[i]))  # [1:-2])
            else:
                singletons = frequent_item_sets[i]
                for j, frequent_singleton in enumerate(singletons):
                    if j != singletons.__len__() - 1:
                        f.write('(\'' + str(frequent_singleton) + '\'),')
                    else:
                        f.write('(\'' + str(frequent_singleton) + '\')\n')
    else:

        for i in range(1, item_set_size):
            if i == 2:
                f.write('\n')
            if i != 1:
                f.write(str(frequent_item_set_candidates[i])[1:-1].replace('), ', '),') + '\n\n')

            else:
                singletons = frequent_item_set_candidates[i]
                for j, frequent_singleton in enumerate(singletons):
                    if j != singletons.__len__() - 1:
                        f.write('(\'' + str(frequent_singleton) + '\'),')
                    else:
                        f.write('(\'' + str(frequent_singleton) + '\')\n')

        f.write('Frequent Itemsets:\n')
        for i in range(1, item_set_size):
            if i == 2:
                f.write('\n')
            if i != 1:
                if i != item_set_size - 1:
                    f.write(str(frequent_item_sets[i])[1:-1].replace('), ', '),') + '\n\n')
                else:
                    f.write(str(frequent_item_sets[i])[1:-1].replace('),', '),'))
            else:
                singletons = frequent_item_sets[i]
                for j, frequent_singleton in enumerate(singletons):
                    if j != singletons.__len__() - 1:
                        f.write('(\'' + str(frequent_singleton) + '\'),')
                    else:
                        f.write('(\'' + str(frequent_singleton) + '\')\n')

    f.close()


def main():
    sc, baskets, sample_support_vars = set_up_task1()
    # declare solution dictionaries
    frequent_item_set_candidates, frequent_item_sets = declare_son_vars()

    # track item set size... increases by 1 per iteration
    item_set_size = 1
    keep_looping = True

    while keep_looping:
        if item_set_size > 1:
            previous_frequent_item_sets = frequent_item_sets[item_set_size - 1]
            if previous_frequent_item_sets.__len__() == 0:
                keep_looping = False
                # print('breakpoint_1')
                # uncomment break while looping
                break
        else:
            previous_frequent_item_sets = None

        frequent_item_set_candidates[item_set_size] = tuple(sorted(
            baskets.mapPartitions(
                lambda iterator: phase_one(iterator, sample_support_vars, previous_frequent_item_sets, item_set_size)).
                flatMap(lambda element: element).distinct().collect()))

        if frequent_item_set_candidates[item_set_size].__len__() == 0:
            keep_looping = False
            del frequent_item_set_candidates[item_set_size]
            # print('No more item set candidates')
            # uncomment break while looping
            break

        if item_set_size == 1:
            baskets = baskets.map(lambda basket: prune_baskets(basket, frequent_item_set_candidates[item_set_size])).repartition(3).persist()

        frequent_item_sets[item_set_size] = tuple(sorted(
            baskets.mapPartitions(lambda iterator: phase_two(iterator, frequent_item_set_candidates[item_set_size], item_set_size)).
                flatMap(lambda element: element).reduceByKey(add).
                filter(lambda element: element[1] >= sample_support_vars['support']).
                map(lambda element: element[0]).collect()))

        if item_set_size == 1:
            baskets = baskets.map(lambda basket: prune_baskets(basket, frequent_item_sets[item_set_size])).repartition(3).persist()

        if frequent_item_sets[item_set_size].__len__() == 0:
            keep_looping = False
            del frequent_item_sets[item_set_size]
            # print('breakpoint_3')
            # uncomment break while looping
            break

        item_set_size += 1

    write_output_file(item_set_size, frequent_item_set_candidates, frequent_item_sets)

if __name__ == '__main__':
    main()