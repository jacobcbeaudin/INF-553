from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext
from itertools import islice
from csv import writer
from sys import argv


def task2_case1():
    def start_spark_context():
        sc = SparkContext('local[*]', 'hw3_task_1')
        sc.setLogLevel("OFF")
        return sc

    def read_train_val():
        data_raw = dict()
        data_raw['train'] = sc.textFile(argv[1]).\
            mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element).\
            map(lambda element: element.split(',')).map(
            lambda element: (element[0], element[1], float(element[2]))).repartition(1)

        data_raw['val'] = sc.textFile(argv[2]).\
            mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element).\
            map(lambda element: element.split(',')).map(
            lambda element: (element[0], element[1], float(element[2]))).repartition(1)

        return data_raw

    def convert_data(data_raw):

        id_codes = (data_raw['train'] + data_raw['val']).repartition(1).flatMap(
            lambda element: ([element[0], element[1]])).distinct().collect()

        id_lookup = dict()

        for id_key, id_code in enumerate(id_codes):
            id_lookup[id_code] = id_key
            id_lookup[id_key] = id_code

        data_converted = dict()

        data_converted['train'] = data_raw['train'].map(
            lambda element: Rating(id_lookup[element[0]], id_lookup[element[1]], element[2])).persist()

        data_converted['val'] = data_raw['val'].map(
            lambda element: Rating(id_lookup[element[0]], id_lookup[element[1]], element[2]))

        data_converted['val_data_count'] = data_converted['val'].count()
        # data_converted['train_data_count'] = data_converted['train'].count()

        return data_converted, id_lookup

    def group_user_business_from_rating(element):
        return tuple([tuple([element[0], element[1]]), element[2]])

    sc = start_spark_context()

    data_raw = read_train_val()
    # cold start use subtractByKey
    data_converted, id_lookup = convert_data(data_raw)

    # iterations good at 4, better as increases
    model = ALS.train(data_converted['train'], rank=4, iterations=10, lambda_=0.2)

    val_predictions = model.predictAll(data_converted['val'].map(lambda element: (element[0], element[1]))).map(
        lambda element: tuple([element[0], element[1], element[2]])).repartition(1)

    val_cold_start_predictions = data_converted['val'].map(
        lambda element: ((element[0], element[1]), 1)).subtractByKey(
        val_predictions.map(
            lambda element: ((element[0], element[1]), 1))).repartition(1).map(
        lambda element: (element[0][0], element[0][1], 3.7))

    val_predictions = (val_predictions + val_cold_start_predictions).repartition(1).map(
        lambda element: (id_lookup[element[0]], id_lookup[element[1]], element[2])).repartition(1).collect()

    with open(argv[4], 'w', newline='') as csvfile:
        spamwriter = writer(csvfile)
        spamwriter.writerow(('user_id', 'business_id', 'rating'))
        spamwriter.writerows(val_predictions)


def task2_case2():

    # if business or user does not exist try ignore OR set all other values to number between 2 and 3... recommends 3.5!!

    def start_spark_context():
        sc = SparkContext('local[*]', 'hw3_task_1')
        sc.setLogLevel("OFF")
        return sc

    def read_train_val():
        data_raw = dict()

        data_raw['train'] = sc.textFile(argv[1]).\
            mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element). \
            map(lambda element: element.split(',')).map(
            lambda element: (element[0], element[1], float(element[2])))

        data_raw['val'] = sc.textFile(argv[2]).\
            mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element). \
            map(lambda element: element.split(',')).map(
            lambda element: (element[0], element[1], float(element[2])))

        return data_raw

    def get_train_lookup(train_data):
        train_lookup = dict()
        train_lookup['user'] = train_data.map(lambda element: tuple([element[0], element[1]])).groupByKey().map(
            lambda element: tuple([element[0], set(element[1])])).collectAsMap()

        train_lookup['business'] = train_data.map(lambda element: tuple([element[1], element[0]])).groupByKey().map(
            lambda element: tuple([element[0], set(element[1])])).collectAsMap()

        train_lookup['ratings'] = train_data.map(
            lambda element: tuple([tuple([element[0], element[1]]), element[2]])).collectAsMap()

        return train_lookup

    def get_user_ratings_lookup(train_data):
        user_ratings_lookup = train_data.map(lambda element: tuple([element[0], element[2]])).groupByKey().map(
            lambda element: (element[0], (sum(element[1]), element[1].__len__()))).collectAsMap()
        return user_ratings_lookup

    def get_user_based_prediction(predicted_user, predicted_business, pearson_correlations):
        avg_ratings = dict()

        avg_ratings[predicted_user] = user_ratings_lookup[predicted_user][0] / user_ratings_lookup[predicted_user][1]

        # print(pearson_correlations)
        for nearest_neighbor in pearson_correlations.keys():
            avg_ratings[nearest_neighbor] = (user_ratings_lookup[nearest_neighbor][0] -
                                             train_lookup['ratings'][(nearest_neighbor, predicted_business)]) / \
                                            (user_ratings_lookup[nearest_neighbor][1] - 1)

        prediction_components = dict()
        prediction_components['numerator'] = 0

        prediction_components['denominator'] = sum(map(abs, pearson_correlations.values()))

        for nearest_neighbor in pearson_correlations.keys():
            prediction_components['numerator'] += (train_lookup['ratings'][(nearest_neighbor, predicted_business)] -
                                                   avg_ratings[nearest_neighbor]) * pearson_correlations[
                                                      nearest_neighbor]

        predicted_rating = avg_ratings[predicted_user] + \
                           prediction_components['numerator'] / prediction_components['denominator']

        return predicted_rating

    def user_based_predictions(predicted_user, predicted_business):
        # check if predicted_user exists in training set
        if predicted_user in train_lookup['user']:
            # check if predicted_business exists in training set
            if predicted_business in train_lookup['business']:

                # make dictionary of pearson correlations of predicted user and nearest neighbor
                pearson_correlations = dict()
                # possible nearest neighbors... all users who rated predicted_business EXCLUDING predicted_user
                possible_nns = tuple(train_lookup['business'][predicted_business].difference(set(predicted_user)))

                # check if there is more than one possible nearest neighbor
                if possible_nns.__len__() > 1:
                    for nn_candidate in possible_nns:

                        # check if pearson correlation has been calculated between predicted_user & nn_candidate... [idea] try: / except KeyError:
                        if (predicted_user, nn_candidate) not in user_pairs_lookup:
                            # add predicted_user, nn_candidate to dictionary
                            user_pairs_lookup[(predicted_user, nn_candidate)] = dict()

                            # get all business rated by both predicted_user and nn_candidate... INCLUDING predicted_business
                            user_pairs_lookup[(predicted_user, nn_candidate)]['businesses'] = \
                                train_lookup['user'][nn_candidate].intersection(train_lookup['user'][predicted_user])

                            # check if predicted_user & nn_candidate have co-rated at least 2 businesses in addition to predicted_business
                            if user_pairs_lookup[(predicted_user, nn_candidate)]['businesses'].__len__() > 2:
                                # print(user_pairs_lookup[(predicted_user, nn_candidate)]['businesses'].__len__() > 2)

                                # add average value dict
                                user_pairs_lookup[(predicted_user, nn_candidate)]['avg'] = dict()
                                # REMOVE PREDICTED BUSINESS
                                co_rated_businesses_excluding_predicted_business = tuple(user_pairs_lookup[
                                                                                             (predicted_user,
                                                                                              nn_candidate)][
                                                                                             'businesses'].difference(
                                    predicted_business))

                                for user_id in (predicted_user, nn_candidate):
                                    user_ratings_list = [train_lookup['ratings'][(user_id, co_rated_business)] for
                                                         co_rated_business in
                                                         co_rated_businesses_excluding_predicted_business]

                                    user_pairs_lookup[(predicted_user, nn_candidate)]['avg'][user_id] = \
                                        sum(user_ratings_list) / \
                                        co_rated_businesses_excluding_predicted_business.__len__()

                                # CHECK NUMERATOR!!!!!!!!!!!!!!!!!!!!!

                                pearson_numerator = sum([(train_lookup['ratings'][(predicted_user, co_rated_business)] -
                                                          user_pairs_lookup[(predicted_user, nn_candidate)]['avg']
                                                          [predicted_user]) *
                                                         (train_lookup['ratings'][(nn_candidate, co_rated_business)] -
                                                          user_pairs_lookup[(predicted_user, nn_candidate)]['avg']
                                                          [nn_candidate]) for co_rated_business in
                                                         co_rated_businesses_excluding_predicted_business])
                                # print(pearson_numerator)
                                if pearson_numerator == 0:
                                    user_pairs_lookup[(predicted_user, nn_candidate)]['pearson_correlation'] = None
                                    # add reversed key to dict
                                    user_pairs_lookup[(nn_candidate, predicted_user)] = user_pairs_lookup[
                                        (predicted_user, nn_candidate)]
                                    continue
                                else:
                                    # print('numerator not 0')
                                    pearson_denominator = (sum(
                                        [(train_lookup['ratings'][(predicted_user, co_rated_business)] -
                                          user_pairs_lookup[(predicted_user, nn_candidate)]['avg']
                                          [predicted_user]) ** 2 for co_rated_business in
                                         co_rated_businesses_excluding_predicted_business]) * \
                                                           sum([(train_lookup['ratings'][
                                                                     (nn_candidate, co_rated_business)] -
                                                                 user_pairs_lookup[(predicted_user, nn_candidate)][
                                                                     'avg']
                                                                 [nn_candidate]) ** 2 for co_rated_business in
                                                                co_rated_businesses_excluding_predicted_business])) ** (
                                                                      1 / 2)

                                    pearson_correlation_candidate = pearson_numerator / pearson_denominator

                                    # print(pearson_correlation_candidate)

                                    if pearson_correlation_candidate > 0:
                                        pearson_correlations[nn_candidate] = pearson_correlation_candidate
                                        user_pairs_lookup[(predicted_user, nn_candidate)]['pearson_correlation'] = \
                                        pearson_correlations[nn_candidate]
                                        user_pairs_lookup[(nn_candidate, predicted_user)] = user_pairs_lookup[
                                            (predicted_user, nn_candidate)]

                                    else:
                                        user_pairs_lookup[(predicted_user, nn_candidate)]['pearson_correlation'] = None
                                        # add reversed key to dict
                                        user_pairs_lookup[(nn_candidate, predicted_user)] = user_pairs_lookup[
                                            (predicted_user, nn_candidate)]

                            else:
                                user_pairs_lookup[(predicted_user, nn_candidate)]['pearson_correlation'] = None
                                # add reversed key to dict
                                user_pairs_lookup[(nn_candidate, predicted_user)] = user_pairs_lookup[
                                    (predicted_user, nn_candidate)]
                                continue

                            user_pairs_lookup[(nn_candidate, predicted_user)] = user_pairs_lookup[
                                (predicted_user, nn_candidate)]

                        # if pearson correlation has been calculated between predicted_user & nn_candidate... add pearson_correlation to local pearson_correlation dict with nn as key
                        elif user_pairs_lookup[(predicted_user, nn_candidate)]['pearson_correlation'] is not None:
                            pearson_correlations[nn_candidate] = user_pairs_lookup[(predicted_user, nn_candidate)][
                                'pearson_correlation']
                        else:
                            continue
                    if pearson_correlations.__len__() > 0:
                        predicted_rating = get_user_based_prediction(predicted_user, predicted_business,
                                                                     pearson_correlations)
                        if predicted_rating > 4.5:
                            predicted_rating = 4.5
                        elif predicted_rating < 1.75:
                            predicted_rating = 1.75

                    else:
                        predicted_rating = user_ratings_lookup[predicted_user][0] / user_ratings_lookup[predicted_user][
                            1]
                else:
                    predicted_rating = user_ratings_lookup[predicted_user][0] / user_ratings_lookup[predicted_user][1]
            else:
                predicted_rating = user_ratings_lookup[predicted_user][0] / user_ratings_lookup[predicted_user][1]
        else:
            predicted_rating = 3.72
        return predicted_rating

    try:
        sc = start_spark_context()
    except ValueError:
        pass

    data_raw = read_train_val()
    # users and their businesses
    train_lookup = get_train_lookup(data_raw['train'])

    user_ratings_lookup = get_user_ratings_lookup(data_raw['train'])

    user_pairs_lookup = dict()

    val_predictions = data_raw['val'].map(lambda element: (element[0], element[1], user_based_predictions(element[0], element[1]))).collect()

    with open(argv[4], 'w', newline='') as csvfile:
        spamwriter = writer(csvfile)
        spamwriter.writerow(('user_id', 'business_id', 'rating'))
        spamwriter.writerows(val_predictions)


def task2_case3():
    def start_spark_context():
        sc = SparkContext('local[*]', 'hw3_task_1')
        sc.setLogLevel("OFF")
        return sc

    def read_train_val():
        data_raw = dict()

        data_raw['train'] = sc.textFile(argv[1]).\
            mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element). \
            map(lambda element: element.split(',')).map(
            lambda element: (element[0], element[1], float(element[2])))

        data_raw['val'] = sc.textFile(argv[2]).\
            mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element).\
            map(lambda element: element.split(',')).map(
            lambda element: (element[0], element[1], float(element[2])))

        return data_raw

    def get_train_lookup(train_data):
        train_lookup = dict()
        train_lookup['user'] = train_data.map(lambda element: tuple([element[0], element[1]])).groupByKey().map(
            lambda element: tuple([element[0], set(element[1])])).collectAsMap()

        train_lookup['business'] = train_data.map(lambda element: tuple([element[1], element[0]])).groupByKey().map(
            lambda element: tuple([element[0], set(element[1])])).collectAsMap()

        train_lookup['ratings'] = train_data.map(
            lambda element: tuple([tuple([element[0], element[1]]), element[2]])).collectAsMap()

        return train_lookup

    def get_user_ratings_lookup(train_data):
        # items is 0 and business is 1
        user_ratings_lookup = train_data.map(lambda element: tuple([element[1], element[2]])).groupByKey().map(
            lambda element: (element[0], (sum(element[1]), element[1].__len__()))).collectAsMap()
        return user_ratings_lookup

    def get_business_based_prediction(predicted_user, predicted_business, pearson_correlations):
        prediction_components = dict()
        prediction_components['numerator'] = 0

        prediction_components['denominator'] = sum(map(abs, pearson_correlations.values()))

        for nearest_neighbor in pearson_correlations:
            prediction_components['numerator'] += train_lookup['ratings'][(predicted_user, nearest_neighbor)] * \
                                                  pearson_correlations[nearest_neighbor]

        predicted_rating = prediction_components['numerator'] / prediction_components['denominator']

        return predicted_rating

    def user_based_predictions(predicted_user, predicted_business):
        # check if predicted_business exists in training set
        if predicted_business in train_lookup['business']:

            # check if predicted_user exists in training set
            if predicted_user in train_lookup['user']:

                # make dictionary of pearson correlations of predicted user and nearest neighbor
                pearson_correlations = dict()
                # possible nearest neighbors... all business rated by predicted_user EXCLUDING predicted_business
                possible_nns = tuple(train_lookup['user'][predicted_user].difference(set(predicted_business)))

                # check if there is more than one possible nearest neighbor
                if possible_nns.__len__() > 1:
                    for nn_candidate in possible_nns:
                        # check if pearson correlation has been calculated between predicted_business & nn_candidate... [idea] try: / except KeyError:
                        if (predicted_business, nn_candidate) not in business_pairs_lookup:
                            # add predicted_business, nn_candidate to dictionary
                            business_pairs_lookup[(predicted_business, nn_candidate)] = dict()

                            # get all business rated by both predicted_user and nn_candidate... INCLUDING predicted_business
                            business_pairs_lookup[(predicted_business, nn_candidate)]['users'] = \
                                train_lookup['business'][nn_candidate].intersection(
                                    train_lookup['business'][predicted_business])

                            # check if predicted_user & nn_candidate have co-rated at least 2 businesses in addition to predicted_business
                            if business_pairs_lookup[(predicted_business, nn_candidate)]['users'].__len__() > 2:

                                # add average value dict
                                business_pairs_lookup[(predicted_business, nn_candidate)]['avg'] = dict()
                                # REMOVE PREDICTED BUSINESS
                                co_rating_users_excluding_predicted_user = tuple(business_pairs_lookup[
                                                                                     (
                                                                                     predicted_business, nn_candidate)][
                                                                                     'users'].difference(
                                    predicted_user))

                                for business_id in (predicted_business, nn_candidate):
                                    user_ratings_list = [train_lookup['ratings'][(co_rating_user, business_id)] for
                                                         co_rating_user in co_rating_users_excluding_predicted_user]

                                    business_pairs_lookup[(predicted_business, nn_candidate)]['avg'][business_id] = \
                                        sum(user_ratings_list) / \
                                        co_rating_users_excluding_predicted_user.__len__()

                                # CHECK NUMERATOR!!!!!!!!!!!!!!!!!!!!!

                                pearson_numerator = sum(
                                    [(train_lookup['ratings'][(co_rating_user, predicted_business)] -
                                      business_pairs_lookup[(predicted_business, nn_candidate)]['avg']
                                      [predicted_business]) *
                                     (train_lookup['ratings'][(co_rating_user, nn_candidate)] -
                                      business_pairs_lookup[(predicted_business, nn_candidate)]['avg']
                                      [nn_candidate]) for co_rating_user in
                                     co_rating_users_excluding_predicted_user])
                                # print(pearson_numerator)
                                if pearson_numerator == 0:
                                    business_pairs_lookup[(predicted_business, nn_candidate)][
                                        'pearson_correlation'] = None
                                    # add reversed key to dict
                                    business_pairs_lookup[(nn_candidate, predicted_business)] = business_pairs_lookup[
                                        (predicted_business, nn_candidate)]
                                    continue
                                else:
                                    # print('numerator not 0')
                                    pearson_denominator = (sum(
                                        [(train_lookup['ratings'][(co_rating_user, predicted_business)] -
                                          business_pairs_lookup[(predicted_business, nn_candidate)]['avg']
                                          [predicted_business]) ** 2 for co_rating_user in
                                         co_rating_users_excluding_predicted_user]) * \
                                                           sum([(train_lookup['ratings'][
                                                                     (co_rating_user, nn_candidate)] -
                                                                 business_pairs_lookup[
                                                                     (predicted_business, nn_candidate)]['avg']
                                                                 [nn_candidate]) ** 2 for co_rating_user in
                                                                co_rating_users_excluding_predicted_user])) ** (1 / 2)

                                    pearson_correlation_candidate = pearson_numerator / pearson_denominator

                                    # print(pearson_correlation_candidate)

                                    if pearson_correlation_candidate > 0:
                                        pearson_correlations[nn_candidate] = pearson_correlation_candidate
                                        business_pairs_lookup[(predicted_business, nn_candidate)][
                                            'pearson_correlation'] = pearson_correlations[nn_candidate]
                                        business_pairs_lookup[(nn_candidate, predicted_business)] = \
                                        business_pairs_lookup[
                                            (predicted_business, nn_candidate)]

                                    else:
                                        business_pairs_lookup[(predicted_business, nn_candidate)][
                                            'pearson_correlation'] = None
                                        # add reversed key to dict
                                        business_pairs_lookup[(nn_candidate, predicted_business)] = \
                                        business_pairs_lookup[
                                            (predicted_business, nn_candidate)]

                            else:
                                business_pairs_lookup[(predicted_business, nn_candidate)]['pearson_correlation'] = None
                                # add reversed key to dict
                                business_pairs_lookup[(nn_candidate, predicted_business)] = business_pairs_lookup[
                                    (predicted_business, nn_candidate)]
                                continue

                            business_pairs_lookup[(nn_candidate, predicted_business)] = business_pairs_lookup[
                                (predicted_business, nn_candidate)]

                        # if pearson correlation has been calculated between predicted_user & nn_candidate... add pearson_correlation to local pearson_correlation dict with nn as key
                        elif business_pairs_lookup[(predicted_business, nn_candidate)][
                            'pearson_correlation'] is not None:
                            pearson_correlations[nn_candidate] = \
                            business_pairs_lookup[(predicted_business, nn_candidate)]['pearson_correlation']
                        else:
                            continue
                    if pearson_correlations.__len__() > 12:
                        predicted_rating = get_business_based_prediction(predicted_user, predicted_business,
                                                                         pearson_correlations)
                        if predicted_rating > 4.5:
                            predicted_rating = 4.5
                        elif predicted_rating < 1.75:
                            predicted_rating = 1.75

                    else:
                        predicted_rating = user_ratings_lookup[predicted_business][0] / \
                                           user_ratings_lookup[predicted_business][1]
                else:
                    predicted_rating = user_ratings_lookup[predicted_business][0] / \
                                       user_ratings_lookup[predicted_business][1]
            else:
                predicted_rating = user_ratings_lookup[predicted_business][0] / \
                                   user_ratings_lookup[predicted_business][1]
        else:
            predicted_rating = 3.72
        return predicted_rating

    try:
        sc = start_spark_context()
    except ValueError:
        pass

    data_raw = read_train_val()
    # users and their businesses
    train_lookup = get_train_lookup(data_raw['train'])

    user_ratings_lookup = get_user_ratings_lookup(data_raw['train'])

    business_pairs_lookup = dict()

    val_predictions = data_raw['val'].map(lambda element: (element[0], element[1], user_based_predictions(element[0], element[1]))).collect()

    with open(argv[4], 'w', newline='') as csvfile:
        spamwriter = writer(csvfile)
        spamwriter.writerow(('user_id', 'business_id', 'rating'))
        spamwriter.writerows(val_predictions)


def main():
    if int(argv[3]) == 1:
        task2_case1()
    elif int(argv[3]) == 2:
        task2_case2()
    elif int(argv[3]) == 3:
        task2_case3()
    else:
        print('error not a valid case... select case in {1,2,3}')


if __name__ == '__main__':
    main()
