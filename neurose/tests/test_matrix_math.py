from unittest import TestCase
from neurose.matrix_math import MatrixMath
import numpy as np
from random import randint


class TestMatrixMath(TestCase):

    def test_matrix_dot_with_basic_case(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7,8]]

        self.assertTrue((np.matrix(a) * np.matrix(b) == MatrixMath.dot(a, b)).all())

    def test_matrix_dot_with_big_matrixes(self):
        a = [[randint(-10, 10) for i in range(50)] for i in range(30)]
        b = [[randint(-10, 10) for i in range(30)] for i in range(50)]
        self.assertTrue((np.matrix(a) * np.matrix(b) == MatrixMath.dot(a, b)).all())

    def test_matrix_dot_with_different_dimensions(self):
        a = [[randint(-10, 10) for i in range(3)] for i in range(2)]
        b = [[randint(-10, 10) for i in range(2)] for i in range(3)]
        self.assertTrue((np.matrix(a) * np.matrix(b) == MatrixMath.dot(a, b)).all())

    def test_matrix_dot_with_more_dimensions(self):
        a = [[randint(-10, 10)] for i in range(5)]
        b = [[randint(-10, 10) for i in range(5)]]
        self.assertTrue((np.matrix(a) * np.matrix(b) == MatrixMath.dot(a, b)).all())

    def test_converts_vector_to_matrix(self):
        a = [[randint(-10, 10)] for i in range(5)]
        b = [randint(-10, 10) for i in range(5)]
        self.assertTrue((np.matrix(a) * np.matrix(b) == MatrixMath.dot(a, b)).all())

    def test_converts_vector_to_matrix_2(self):
        b = [[randint(-10, 10)] for i in range(5)]
        a = [randint(-10, 10) for i in range(5)]
        self.assertTrue((np.matrix(a) * np.matrix(b) == MatrixMath.dot(a, b)).all())

    def test_matrix_dot_with_weird_dimensions(self):
        b = [[randint(-10, 10)] for i in range(5)]
        a = [[randint(-10, 10) for i in range(5)]]
        self.assertTrue((np.matrix(a) * np.matrix(b) == MatrixMath.dot(a, b)).all())

    def test_matrix_dot_throws_error_if_dimensions_dont_match(self):
        b = [[1, 2, 1], [3, 4]]
        a = [[5, 6], [7,8]]

        self.assertRaises(ValueError, MatrixMath.dot, a, b)

    def test_matrix_dot_throws_error_if_dimensions_dont_match_2(self):
        b = [[1], [2, 3, 3, 4]]
        a = [[5, 6], [7,8, 7]]

        self.assertRaises(ValueError, MatrixMath.dot, a, b)

    def test_matrix_dot_throws_error_if_dimensions_dont_match_3(self):
        b = [[1], 2, 3, 4, 5]
        a = [[randint(-10, 10) for i in range(5)]]
        self.assertRaises(ValueError, MatrixMath.dot, a, b)

    def test_matrix_dot_throws_error_if_dimensions_dont_match_4(self):
        b = [1, [2], 3, 4, 5]
        a = [[randint(-10, 10) for i in range(5)]]
        self.assertRaises(ValueError, MatrixMath.dot, a, b)

    def test_matrix_dot_matrixes_of_same_asymmetric_shape_raises_error(self):
        a = [[randint(-10, 10) for i in range(2)] for i in range(3)]
        b = [[randint(-10, 10) for i in range(2)] for i in range(3)]
        self.assertRaises(ValueError, MatrixMath.dot, a, b)

    def test_matrix_dot_throws_error_if_dimensions_dont_match_6(self):
        a = [[randint(-10, 10) for i in range(10)] for i in range(30)]
        b = [[randint(-10, 10) for i in range(29)] for i in range(9)]
        self.assertRaises(ValueError, MatrixMath.dot, a, b)

    def test_matrix_works_on_empty_matrixes(self):
        assert MatrixMath.dot([], []) == []
