from pyspark import SparkContext, SparkConf
from itertools import islice, combinations
from random import seed, sample
from sys import argv
from csv import writer


def get_id_dicts(business_user_pairs_train):
    user_ids = business_user_pairs_train.map(lambda element: element[1]).distinct().collect()
    user_id_to_key = dict()

    for i, user_id in enumerate(user_ids):
        user_id_to_key[user_id] = i

    return user_id_to_key


def start_spark_context():
    # conf = SparkConf().setAppName('Model-based').setMaster('local').set('spark.executor.memory', '4g').\
    #     set('spark.driver.memory', '4g')
    # sc = SparkContext('local[*]', 'hw3_task_1', conf=conf)
    sc = SparkContext('local[*]', 'hw3_task_1')

    sc.setLogLevel("OFF")
    return sc


def read_train_data(sc):
    # No need to include ratings. If in file, ratings exist... in form of [user_id, business_id, stars]
    # map values in order of business_id then user_id
    business_user_pairs_train = sc.textFile(argv[1]).\
        mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element).\
        map(lambda element: element.split(',')).map(lambda element: (element[1], element[0])).repartition(1)


    # hash user_ids to integers
    user_id_to_key = get_id_dicts(business_user_pairs_train)

    # convert user_ids to keys
    business_users_train = business_user_pairs_train.map(lambda element: (element[0], user_id_to_key[element[1]])).\
        groupByKey().map(
        lambda element: (element[0], tuple(element[1])) if type(element[1]) != int else (element[0], element[1]))

    hash_function_constants = get_hash_function_constants(user_id_to_key.keys().__len__())

    return hash_function_constants, business_users_train


def get_lsh_parameters():
    lsh_parameters = dict()
    lsh_parameters['bands'] = 20
    lsh_parameters['rows'] = 2
    return lsh_parameters


def get_min_hash_signature(user_rows, hash_function_constants):
    min_hash_signature = []
    if not isinstance(user_rows, int):
        for a in hash_function_constants['a']:
            for b in hash_function_constants['b']:
                min_hash_signature.append(min([(a * user_row + b) % hash_function_constants['m']
                                               for user_row in user_rows]))
    else:
        for a in hash_function_constants['a']:
            for b in hash_function_constants['b']:
                min_hash_signature.append((a * user_rows + b) % hash_function_constants['m'])
    return min_hash_signature


def get_hash_function_constants(n_users):
    hash_function_constants = dict()
    # 30 to 40 hash functions
    rand_prime = {463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
                  601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727,
                  733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859,
                  863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997}

    seed(69)
    hash_function_constants['a'] = tuple(sample(rand_prime, 5))
    rand_prime.difference(set(hash_function_constants['a']))
    hash_function_constants['b'] = tuple(sample(rand_prime, 8))
    hash_function_constants['n_hash_functions'] = hash_function_constants['a'].__len__() * hash_function_constants['b'].__len__()
    hash_function_constants['m'] = n_users

    return hash_function_constants


def locality_sensitive_hashing(element):
    lsh_parameters = get_lsh_parameters()

    if lsh_parameters['bands'] * lsh_parameters['rows'] != hash_function_constants['n_hash_functions']:
        print('bands ' + str(lsh_parameters['bands']) + ' multiplied by rows ' + str(lsh_parameters['rows']) +
              ' does not equal # of hash functions: ' + str(hash_function_constants['n_hash_functions']))

    lsh_bands = []
    for band in range(lsh_parameters['bands']):
        lsh_bands.append(
            tuple([tuple([band] + element[1][band * lsh_parameters['rows']:(band + 1) * lsh_parameters['rows']]),
                   tuple([element[0], element[2]])]))
    return tuple(lsh_bands)


def get_jaccard_similarity(element):
    return tuple([element[0][0], element[1][0],
                  len(element[0][1].intersection(element[1][1])) / len(element[0][1].union(element[1][1]))])


sc = start_spark_context()

hash_function_constants, business_users_train = read_train_data(sc)

min_hash_signatures = business_users_train.map(
    lambda element: (element[0], get_min_hash_signature(
        user_rows=element[1], hash_function_constants=hash_function_constants), frozenset(element[1])))

business_pairs_jaccard_similarity = min_hash_signatures.map(
    lambda element: (locality_sensitive_hashing(element))).flatMap(lambda element: element).groupByKey().\
    map(lambda element: tuple(element[1])).distinct().filter(lambda element: element.__len__() > 1).\
    map(lambda element: sorted(element, key=lambda element: element[0])).map(
    lambda element: combinations(element, 2)).flatMap(lambda element: element).distinct().map(
    lambda element: get_jaccard_similarity(element)).filter(lambda element: element[2] >= 0.5).sortBy(
    lambda element: element[0]).collect()

with open(argv[2], 'w', newline='') as csvfile:
    spamwriter = writer(csvfile)
    spamwriter.writerow(('business_id_1', 'business_id_2', 'similarity'))
    spamwriter.writerows(business_pairs_jaccard_similarity)

