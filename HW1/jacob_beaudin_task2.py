from pyspark import SparkContext
from json import loads, dump
from sys import argv
from time import time


def main():
    sc = SparkContext('local[*]', 'task_2')

    # get rid of red warnings. create a log4j file change settings
    sc.setLogLevel("OFF")

    # coalesce(12) makes 17 partitions... unsure why... map tuples of business ids and stars from review.json
    review = sc.textFile(argv[1]).coalesce(7).map(loads).map(lambda element: (element['business_id'], element['stars']))

    # convert business.json to dictionary... map tuples of business ids and states from business.json
    business = sc.textFile(argv[2]).coalesce(12).map(loads).map(lambda element: (element['business_id'], element['state']))

    # task 2 part a
    state_stars = task_2_part_a(review, business, sc, argv[3])

    # task 2 part b
    task_2_part_b(state_stars.map(lambda element: element[0]), argv[4])


def task_2_part_a(review, business, sc, output_file):
    # A. Wt are the average stars for each state? (DO NOT use the stars information in the business file) (2.5 point)
    # Get average reviews per state
    state_stars = business.join(review).map(lambda element: (element[1][0], element[1][1])). \
        aggregateByKey((0, 0), lambda element1, element2: (element1[0] + element2, element1[1] + 1),
                       lambda element1, element2: (element1[0] + element2[0], element1[1] + element2[1])). \
        mapValues(lambda element: element[0] / element[1]).sortBy(lambda element: (-element[1], element[0]))

    # add state,stars to beginning of RDD and convert to list
    state_stars_list = sc.parallelize(['state,stars']). \
        union(state_stars.map(lambda element_tuple: ','.join(str(element) for element in element_tuple))).collect()

    # Write output file
    f = open(output_file, 'w')
    for line in state_stars_list:
        if line != state_stars_list[-1]:
            f.write(line + '\n')
        else:
            f.write(line)
            f.close()

    return state_stars


def task_2_part_b(state_stars, output_file):
    # B. You are required to use two ways to print top 5 states with highest stars. You need to compare the time
    # difference between two methods and explain the result within 1 or 2 sentences. (3 point)
    # Method1: Collect all the data, and then print the first 5 states
    # Method2: Take the first 5 states, and then print all

    m1 = time()
    print(state_stars.collect()[:5])
    m1 = time() - m1

    m2 = time()
    print(state_stars.take(5))
    m2 = time() - m2

    task_2_part_b_results = dict(m1=m1, m2=m2, reason='The collect function is much slower because it is an action and produces all states and then slices the states with the top 5 average ratings. '
                                                      'The take function is much quicker because it done intermittently and produces only the states with the top 5 average ratings.')

    # write output file
    f = open(output_file, 'w')
    dump(task_2_part_b_results, f, indent=4)
    f.close()


if __name__ == "__main__":
    main()

# python dictionary and set
# scala map / hashmap... both are hashtable
