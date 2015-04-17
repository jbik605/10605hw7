import sys

import numpy
import pyspark
import random
from functools import partial

# Constants
MIN_RANDOM_VALUE = 0  # Min random value for the W and H matrices
MAX_RANDOM_VALUE = 1  # Max random value for the W and H matrices
CALCULATE_LOSS = True  # True iff we want to calculate loss per iteration
CHECKPOINT_ITERATION = 81  # Iteration number to evaluate RDD to avoid overflow
MIN_EPS = 0.01  # Minimum value of eps to make sure the SGD does progress
USE_MIN_EPS = True  # True iff we should enforce the MIN_EPS constraint

def to_triplet(graph_input):
    """
    Converts the graph input (graph_input) (text lien) to a triplet.
    :param graph_input:
    :return:
    """
    array = graph_input.split(',')
    return int(array[0]) - 1, int(array[1]) - 1, int(array[2])


def row_matrix_to_tuples(matrix, size):
    """
    Converts the matrix (matrix) to a list of tuples
    :param matrix: The matrix
    :param size: The number of cols in each row
    :return: A list of tuples
    """
    triplet_list = []
    for row in matrix:
        cols = matrix[row]
        for i in range(size):
            if cols[i] != 0:
                triplet_list.append((row, i, cols[i]))
    return triplet_list


def col_matrix_to_tuples(matrix, size):
    """
    Converts the matrix (matrix) to a list of tuples
    :param matrix: The matrix
    :param size: The number of rows in each column
    :return: A list of tuples
    """
    triplet_list = []
    for col in matrix:
        rows = matrix[col]
        for i in range(size):
            if rows[i] != 0:
                triplet_list.append((i, col, rows[i]))
    return triplet_list


def should_use_cell(row, col, workers, s):
    """
    Returns True iff the cell (row, col) of V should be used for stratum s
    :param row: The row number
    :param col: The column number
    :param workers: The amount of workers
    :param s: The stratum number
    :return: True iff the cell (row, col) of V should be used for stratum s
    """
    row = row % workers
    col = col % workers
    s = s % workers
    if row > col:
        return workers - row + col == s
    else:
        return col - row == s


def write_rdd(rdd, dim_row, dim_col, output_file_path):
    """
    Writes the rdd (rdd) in the file with path (output_file_path).
    It is assumed that the rdd represents a matrix of size (dim_row x dim_col)

    Assumed that the rdd fits in memory. This is not ideal, but the RDD requires
    this format to run, unfortunately.

    :param rdd: The rdd to write
    :param dim_row: The number of rows in the matrix contained in the rdd
    :param dim_col: The number of cols in the matrix contained in the rdd
    :param output_file_path: The output file where the rdd contents will be saved
    :return: Nothing
    """
    matrix = numpy.zeros((dim_row, dim_col))

    for w_elem in rdd.collect():
        matrix[w_elem[0], w_elem[1]] = w_elem[2]

    numpy.savetxt(output_file_path, matrix, delimiter=',')


def random_matrix_generator(rows, cols, min_value, max_value):
    """
    Generates a random matrix with rows (rows) and cols (cols. Each element
    will be in the range [min_value, max_value].
    :param rows: Number of rows in the matrix
    :param cols: Number of cols in the matrix
    :param min_value: Min value of each entry
    :param max_value: Max value of each entry
    :return: A generator that generates all of the elements in the matrix
    """
    for i in range(rows):
        for j in range(cols):
            yield (i, j, random.uniform(min_value, max_value))


def write_list(list_to_write, path):
    """
    Writes the list (list_to_write) to the file with path (path).
    One line per list element
    :param list_to_write: The list to write
    :param path: Path of the file
    :return: Nothing
    """
    with open(path, 'w+') as new_file:
        for x in list_to_write:
            new_file.write(str(x) + "\n")


def perform_sgd(iterator, n_i_obj, n_j_obj, beta_val_obj, lambda_val_obj, num_factors_obj, num_updates_obj):
    """
    Performs the sgd updates on a single stratum.
    :param iterator: Iterator to the V, W and H matrices.
    :param n_i_obj: Indicates how many users the user i has rated
    :param n_j_obj: Indicates how many times the movie j has been rated by users
    :param beta_val_obj: The value of beta
    :param lambda_val_obj: The value of lambda
    :param num_factors_obj: The number of latent factors
    :param num_updates_obj: The number of previous DSGD updates done
    :return: The updated W and H matrices for this block
    """
    n_i = n_i_obj.value
    n_j = n_j_obj.value
    beta_val = beta_val_obj.value
    lambda_val = lambda_val_obj.value
    num_factors = num_factors_obj.value
    previous_updates = num_updates_obj.value

    tuples = iterator.next()
    worker_id, iterators = tuples[0], tuples[1]
    v_iterator = iterators[0]
    w_iterator = iterators[1]
    h_iterator = iterators[2]

    sgd_w_mat = {}
    sgd_h_mat = {}
    for row, col, val in w_iterator:
        if not row in sgd_w_mat:
            sgd_w_mat[row] = numpy.zeros(num_factors)
        sgd_w_mat[row][col] = val

    for row, col, val in h_iterator:
        if not col in sgd_h_mat:
            sgd_h_mat[col] = numpy.zeros(num_factors)
        sgd_h_mat[col][row] = val

    updates_done = 0
    t_0 = 1000
    for row, col, rating in v_iterator:
        eps = (t_0 + (updates_done+previous_updates)) ** (-beta_val)
        if USE_MIN_EPS:
            eps = max(MIN_EPS, eps)
        updates_done += 1
        w_row = sgd_w_mat[row]
        h_col = sgd_h_mat[col]

        dot_product = numpy.inner(w_row, h_col)
        common = 2.0 * (rating - dot_product)

        sgd_w_mat[row] = w_row * (1 - 2 * eps * lambda_val / n_i[row]) + eps * common * h_col
        sgd_h_mat[col] = h_col * (1 - 2 * eps * lambda_val / n_j[col]) + eps * common * w_row
    return (0, row_matrix_to_tuples(sgd_w_mat, num_factors)), (1, col_matrix_to_tuples(sgd_h_mat, num_factors))


def calculate_loss(iterator, num_factors_obj):
    """
    Calculates the loss function for the current state of V, W and H
    :param iterator: The iterator to the V, W and H matrices
    :param num_factors_obj: The number of latent factors used
    :return: The loss in this block of V, W and H
    """
    num_factors = num_factors_obj.value

    tuples = iterator.next()
    worker_id, iterators = tuples[0], tuples[1]
    v_iterator = iterators[0]
    w_iterator = iterators[1]
    h_iterator = iterators[2]

    sgd_w_mat = {}
    sgd_h_mat = {}
    for row, col, val in w_iterator:
        if not row in sgd_w_mat:
            sgd_w_mat[row] = numpy.zeros(num_factors)
        sgd_w_mat[row][col] = val

    for row, col, val in h_iterator:
        if not col in sgd_h_mat:
            sgd_h_mat[col] = numpy.zeros(num_factors)
        sgd_h_mat[col][row] = val

    loss = 0.0
    for row, col, rating in v_iterator:
        w_row = sgd_w_mat[row]
        h_col = sgd_h_mat[col]

        dot_product = numpy.inner(w_row, h_col)
        loss += (rating - dot_product) * (rating - dot_product)
    return [loss]


def map_to_sequential_ids(user_id_rdd, movie_id_rdd):
    """
    Map the user ids and movie ids to sequential ids
    :param user_id_rdd: The rdd containing all of the user ids
    :param movie_id_rdd: The rdd containing all of the movie ids
    :return: A tuple of (user id to seq, max user id, movie id to seq, max movie id)
    """
    user_id_to_seq = {}
    current_seq_user_id = 0
    movie_id_to_seq = {}
    current_seq_movie_id = 0

    for user_id in user_id_rdd.collect():
        user_id_to_seq[user_id] = current_seq_user_id
        current_seq_user_id += 1

    for movie_id in movie_id_rdd.collect():
        movie_id_to_seq[movie_id] = current_seq_movie_id
        current_seq_movie_id += 1

    return user_id_to_seq, current_seq_user_id, movie_id_to_seq, current_seq_movie_id


def run(num_factors, num_workers, num_iterations, beta_val, lambda_val, v_path, w_path, h_path, loss_path):
    """
    Runs the main DSGD function
    :param num_factors: Number of latent factors of W and H
    :param num_workers: Number of workers we're using
    :param num_iterations: Number of iterations to run DSGD for
    :param beta_val: The value of beta
    :param lambda_val: The value of lambda
    :param v_path: Input path for the V matrix
    :param w_path: Output path for the W matrix
    :param h_path: Output path for the H matrix
    :param loss_path: Output path for the loss function
    :return: Nothing
    """

    #Initialize the spark context
    conf = pyspark.SparkConf().setAppName("SGD").setMaster("local[{0}]".format(num_workers))
    sc = pyspark.SparkContext(conf=conf)

    # get triplets (user id, movie id, rating) rdd from the input text file
    raw = sc.textFile(v_path)
    triplets = raw.map(lambda x: to_triplet(x))

    # get all of the distinct user ids and movie ids to map ids to sequential ids
    user_ids = triplets.map(lambda x: x[0]).distinct().sortBy(lambda x: x)
    movie_ids = triplets.map(lambda x: x[1]).distinct().sortBy(lambda x: x)
    user_id_to_seq, max_user, movie_id_to_seq, max_movies = map_to_sequential_ids(user_ids, movie_ids)

    # map triplets to use the sequential user and movie ids
    v = triplets.map(lambda x: (user_id_to_seq[x[0]], movie_id_to_seq[x[1]], x[2]))

    # get N_i and N_j for all of the user ids
    n_i = v.countByKey()
    n_j = v.map(lambda x: (x[1], x[2])).countByKey()

    # broadcast the necessary state to all workers
    n_i_obj = sc.broadcast(n_i)
    n_j_obj = sc.broadcast(n_j)
    beta_obj = sc.broadcast(beta_val)
    lambda_obj = sc.broadcast(lambda_val)
    num_factors_obj = sc.broadcast(num_factors)

    # num updates done at each stratum
    num_updates = sum(n_i.values()) / num_workers

    # construct the v, w and h rdd's
    v_by_key = v.keyBy(lambda x: x[0] % num_workers)
    w = sc.parallelize(random_matrix_generator(max_user, num_factors, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                       num_workers)
    h = sc.parallelize(random_matrix_generator(num_factors, max_movies, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                       num_workers)
    w_by_key = None
    h_by_key = None

    # construct the filtered_v_list rdds for each stratum
    filtered_v_list = []
    for stratum in range(num_workers):
        filtered_v_list.append(v_by_key.filter(lambda x: should_use_cell(x[0], x[1][1], num_workers, stratum))
                               .persist())

    # run the iterations of DSGD
    loss_list = []
    current_iteration = 0
    num_updates_done = 0
    while current_iteration < num_iterations:
        num_updates_obj = sc.broadcast(num_updates_done)
        for stratum in range(num_workers):
            w_by_key = w.keyBy(lambda x: (x[0]) % num_workers)
            h_by_key = h.keyBy(lambda x: (x[1]-stratum) % num_workers)

            ret_rdd = filtered_v_list[stratum].groupWith(w_by_key, h_by_key).partitionBy(num_workers).mapPartitions(
                partial(perform_sgd,
                        n_i_obj=n_i_obj,
                        n_j_obj=n_j_obj,
                        beta_val_obj=beta_obj,
                        lambda_val_obj=lambda_obj,
                        num_factors_obj=num_factors_obj,
                        num_updates_obj=num_updates_obj))

            w = ret_rdd.filter(lambda x: x[0] == 0).map(lambda x: x[1]).flatMap(lambda x: x)
            h = ret_rdd.filter(lambda x: x[0] == 1).map(lambda x: x[1]).flatMap(lambda x: x)
        num_updates_done += num_updates

        if CALCULATE_LOSS:
            # Go over the stratums again to calculate the loss
            loss = 0.0
            for stratum in range(num_workers):
                w_by_key = w.keyBy(lambda x: (x[0]) % num_workers)
                h_by_key = h.keyBy(lambda x: (x[1]-stratum) % num_workers)

                loss_rdd = filtered_v_list[stratum]\
                    .groupWith(w_by_key, h_by_key).partitionBy(num_workers).mapPartitions(
                        partial(calculate_loss, num_factors_obj=num_factors_obj))

                loss += loss_rdd.reduce(lambda a, b: a + b)
            loss_list.append(loss)
        elif current_iteration >= CHECKPOINT_ITERATION and current_iteration % CHECKPOINT_ITERATION == 0:
            w.count()  # fix stack size problem
        current_iteration += 1

    # get final results and write to disk
    final_w = w_by_key.map(lambda x: x[1])
    final_h = h_by_key.map(lambda x: x[1])

    write_rdd(final_w, max_user, num_factors, w_path)
    write_rdd(final_h, num_factors, max_movies, h_path)
    if CALCULATE_LOSS:
        write_list(loss_list, loss_path)

if __name__ == '__main__':
    num_factors_arg = int(sys.argv[1])
    num_workers_arg = int(sys.argv[2])
    num_iterations_arg = int(sys.argv[3])
    beta_val_arg = float(sys.argv[4])
    lambda_val_arg = float(sys.argv[5])
    v_path_arg = sys.argv[6]
    w_path_arg = sys.argv[7]
    h_path_arg = sys.argv[8]
    loss_path_arg = 'loss.txt'
    if len(sys.argv) >= 10:
        loss_path_arg = sys.argv[9]
    run(num_factors_arg, num_workers_arg, num_iterations_arg, beta_val_arg, lambda_val_arg,
        v_path_arg, w_path_arg, h_path_arg, loss_path_arg)


