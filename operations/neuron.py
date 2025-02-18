import csv
import numpy as np


def open_csv(csv_path, tolerance, eta, iterations):
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=";", quotechar="|")
        next(reader)
        pre_matrix = np.array([row for row in reader]).astype(float)
        matrix = np.insert(pre_matrix, 0, 1, axis=1)
        yd = np.array(matrix[:, -1]).astype(int).reshape(-1, 1)
        norm_e_by_iterations = []
        w_by_iterations = []
        w = initialize_w(matrix)
        w_by_iterations.append(w)
        yc_by_iterations = []

        for _ in range(int(iterations)):
            u = get_u(w, matrix)
            yc = activation_function(u)
            yc_by_iterations.append(yc)
            e = obtain_error(yd, yc)
            delta_w = calculate_delta_w(eta, e, matrix)
            norm_e = obtain_norm_e(e)
            norm_e_by_iterations.append(norm_e)
            w = update_w(w, delta_w)
            w_by_iterations.append(w)
            if tolerance and norm_e <= float(tolerance):
                break

        return w_by_iterations, norm_e_by_iterations, yd, yc_by_iterations


def initialize_w(matrix):
    rng = np.random.default_rng()
    return rng.random((1, matrix.shape[1] - 1))


def get_u(w, matrix):
    x = matrix[:, : (matrix.shape[1] - 1)]
    return np.dot(x, w.T)


def activation_function(u):
    return np.where(u >= 0, 1, 0).astype(int)


def obtain_error(yd, yc):
    return yd - yc


def obtain_norm_e(e):
    return np.linalg.norm(e)


def calculate_delta_w(eta, e, matrix):
    x = matrix[:, : (matrix.shape[1] - 1)]
    return eta * np.dot(e.T, x)


def update_w(w, delta_w):
    return w + delta_w
