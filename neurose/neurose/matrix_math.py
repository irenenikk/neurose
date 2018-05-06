class MatrixMath:

    @staticmethod
    def vector_as_matrix(v):
        return [v]

    @staticmethod
    def raise_value_error(m1, m2):
        raise ValueError('Incompatible dimensions: {} dot {}'.format(np.asarray(m1).shape, np.asarray(m2).shape))

    @staticmethod
    def dot(matrix1, matrix2):
        """
        Matrix multiplication.
        :return: The dot product of matrix1 and matrix2.
        """
        if len(matrix1) == 0 or len(matrix2) == 0:
            return []
        # the matrices can be implicitly changed from a single vector to matrix
        if not isinstance(matrix2[0], list):
            matrix2 = MatrixMath.vector_as_matrix(matrix2)
        if not isinstance(matrix1[0], list):
            matrix1 = MatrixMath.vector_as_matrix(matrix1)
        # initialize result matrix with Nones
        result = [None] * len(matrix1)
        for i in range(len(result)):
            result[i] = [None] * len(matrix2[0])
        # start multiplying
        for row in range(len(matrix1)):
            for col in range(len(matrix2[0])):
                sum = 0
                # amount of cols in first matrix must be same as amount of rows in second
                if len(matrix1[row]) != len(matrix2):
                    MatrixMath.raise_value_error(matrix1, matrix2)
                # i is column depth
                for i in range(len(matrix1[row])):
                    a = matrix1[row][i]
                    if not isinstance(matrix2[i], list)\
                        or col >= len(matrix2[i]):
                        MatrixMath.raise_value_error(matrix1, matrix2)
                    b = matrix2[i][col]
                    sum += a*b
                result[row][col] = sum
        return result