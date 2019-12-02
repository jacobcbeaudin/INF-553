from pyspark import SparkContext
from json import loads, dump
from operator import add
from sys import argv


def main():
    sc = SparkContext('local[*]', 'task_1')

    # get rid of red warnings. create a log4j file change settings
    sc.setLogLevel("OFF")

    # load review.json as string then convert to dict
    review = sc.textFile(argv[1]).coalesce(12).map(loads)

    # solve task1 parts a through g
    n_review_useful = task_1_part_a(review)
    n_review_5_star = task_1_part_b(review)
    n_characters = task_1_part_c(review)
    n_user = task_1_part_d(review)
    top20_user = task_1_part_e(review)
    n_business = task_1_part_f(review)
    top20_business = task_1_part_g(review)

    task_1_results = dict(n_review_useful=n_review_useful, n_review_5_star=n_review_5_star,
                          n_characters=n_characters, n_user=n_user, top20_user=top20_user,
                          n_business=n_business, top20_business=top20_business)

    # Write output file
    f = open(argv[2], 'w')
    dump(task_1_results, f, indent=4)
    f.close()


def task_1_part_a(review):
    # A. The number of reviews that people think are useful (The value of tag ‘useful’ > 0) (1 point)
    n_review_useful = review.map(lambda element: element['useful']).filter(lambda element: element > 0).count()
    # n_review_useful = review.filter(lambda element: element['useful'] > 0).count()

    return n_review_useful


def task_1_part_b(review):
    # B. The number of reviews that have 5.0 stars rating (1 point)
    n_review_5_star = review.map(lambda element: element['stars']).filter(lambda element: element == 5.0).count()

    return n_review_5_star


def task_1_part_c(review):
    # C. How many characters are there in the ‘text’ of the longest review (1 point)
    n_characters = review.map(lambda element: len(element['text'])).max()

    return n_characters


def task_1_part_d(review):
    # D. The number of distinct users who wrote reviews (1 point)
    n_user = review.map(lambda element: (element['user_id'], 1)).groupByKey().count()

    return n_user


def task_1_part_e(review):
    # E. The top 20 users who wrote the largest numbers of reviews and the number of reviews they wrote (1 point)
    # If two user_ids have the same number of reviews, sort the user_ids in the alphabetical order
    # take ordered
    top20_user = review.map(lambda element: (element['user_id'], 1)).reduceByKey(add).\
        sortBy(lambda element: (-element[1], element[0])).take(20)

    return top20_user


def task_1_part_f(review):
    # F. The number of distinct businesses that have been reviewed (1 point)
    n_business = review.map(lambda element: (element['business_id'], 1)).groupByKey().count()

    return n_business


def task_1_part_g(review):
    # G. The top 20 businesses that had the largest numbers of reviews and the number of reviews they had (1 point)
    # If two business_ids have the same number of reviews, sort the business_ids in the alphabetical order
    top20_business = review.map(lambda element: (element['business_id'], 1)).reduceByKey(add).\
        sortBy(lambda element: (-element[1], element[0])).take(20)

    return top20_business


if __name__ == "__main__":
    main()

# python dictionary and set
# scala map / hashmap... both are hashtable
