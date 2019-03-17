# -*- coding: utf-8 -*-

import random
import sys
from math import sqrt
import numpy as np
from copy import deepcopy

class Norm:
    @classmethod
    def isVector(cls, matrix):
        return (isinstance(matrix, Matrix) and matrix.csize == 1) or not (isinstance(matrix, Matrix))

    @classmethod
    def norm1(cls, matrix):
        # matrix is actually vector
        if Norm.isVector(matrix):
            return sum(map(abs, matrix))
        elif matrix.csize == matrix.rsize:
            return max(map(sum, map(lambda lis: map(abs, lis), list(zip(*matrix)))))

    @classmethod
    def norm2(cls, matrix):
        if Norm.isVector(matrix):
            return sqrt(sum(map(lambda x: x**2, map(abs, matrix))))
        else:
            return max(map(abs, matrix.eigenValues()))

    @classmethod
    def norm8(cls, matrix):
        if Norm.isVector(matrix):
            return max(map(abs, matrix))
        elif matrix.csize == matrix.rsize:
            return max(map(sum, map(lambda lis: map(abs, lis), matrix)))


class MatrixError(Exception):
    pass


class Matrix(object):
    def __init__(self, r, c):
        self.rows = [[0] * c for x in range(r)]
        self.rsize = r
        self.csize = c

    def __getitem__(self, idx):
        return self.rows[idx]

    def __setitem__(self, idx, item):
        self.rows[idx] = item

    def __str__(self):
        s = '\n'.join([' '.join(["{0:15.10f}".format(item) for item in row]) for row in self.rows])
        return s + '\n'

    def __repr__(self):
        s = str(self.rows)
        rank = str(self.getRank())
        rep = "Matrix: \"%s\", rank: \"%s\"" % (s, rank)
        return rep

    def reset(self):
        """ Reset the matrix data """
        self.rows = [[] for x in range(self.rsize)]

    def transpose(self):
        """ Transpose the matrix. Changes the current matrix """
        self.rsize, self.csize = self.csize, self.rsize
        self.rows = [list(item) for item in zip(*self.rows)]

    def getTranspose(self):
        """ Return a transpose of the matrix without modifying the matrix itself """
        m, n = self.csize, self.rsize
        mat = Matrix(m, n)
        mat.rows = [list(item) for item in zip(*self.rows)]
        return mat

    def getRank(self):
        return (self.rsize, self.csize)

    def eigenValues(self):
        return np.linalg.eigvals(np.array(self.rows, dtype=float)).tolist()

    def det(self):
        return float(np.linalg.det(np.array(self.rows, dtype=float)))

    def reverse(self):
        return Matrix.fromList(np.linalg.inv(np.array(self.rows, dtype=float)).tolist())

    def cond(self):
        return Norm.norm8(self.reverse()) * Norm.norm8(self)

    def inv(self):
        return Matrix.fromList(np.linalg.inv(self.rows).tolist())

    def __eq__(self, mat):
        """ Test equality """
        return (mat.rows == self.rows)

    def __add__(self, mat):
        """ Add a matrix to this matrix and return the new matrix """

        if self.getRank() != mat.getRank():
            raise MatrixError("Trying to add matrixes of varying rank!")

        ret = Matrix(self.rsize, self.csize)
        for x in range(self.rsize):
            row = [sum(item) for item in zip(self.rows[x], mat[x])]
            ret[x] = row

        return ret

    def __sub__(self, mat):
        """ Subtract a matrix from this matrix and return the new matrix """

        if self.getRank() != mat.getRank():
            raise MatrixError("Trying to add matrixes of varying rank!")

        ret = Matrix(self.rsize, self.csize)
        for x in range(self.rsize):
            row = [item[0] - item[1] for item in zip(self.rows[x], mat[x])]
            ret[x] = row

        return ret

    def __mul__(self, mat):
        """ Multiple a matrix with this matrix and return the new matrix """

        if isinstance(mat, int) or isinstance(mat, float):
            C = mat
            ret = Matrix(self.rsize, self.csize)
            ret.rows = [[i*C for i in row] for row in self.rows]
            return ret

        if Norm.isVector(mat):
            #return Matrix.fromList((np.array(self.rows) @ np.array(mat)).tolist())
            return Matrix.fromList((np.array(self.rows) @ np.array(mat.rows)).tolist())

        matm, matn = mat.getRank()

        if (self.csize != matm):
            raise MatrixError("Matrices cannot be multipled!")

        mat_t = mat.getTranspose()
        mulmat = Matrix(self.rsize, matn)

        for x in range(self.rsize):
            for y in range(mat_t.rsize):
                mulmat[x][y] = sum([item[0] * item[1] for item in zip(self.rows[x], mat_t[y])])

        return mulmat
        
    def toTexForm(self):
        res="\\begin{pmatrix}"       
        for row in self.rows:
            for y in range(self.csize):
                res += "{0:15.3f}".format(row[y]) + "&"
            res += "\\\\ \n"
        return res + "\\end{pmatrix}" ; 
    
    def replaceColumn (self, column, i):
        j = 0
        for row in self.rows:
            row[i] = column[j]
            j += 1
     
    def getColumn (self, i):
        res =[];
        for row in self.rows:
            res.append(row[i])
        return res
                
            
                
        
            

    # def __iadd__(self, mat):
    #     """ Add a matrix to this matrix. This modifies the current matrix """
    #     # Calls __add__
    #     tempmat = self + mat
    #     self.rows = tempmat.rows[:]
    #     return self

    # def __isub__(self, mat):
    #     """ Add a matrix to this matrix. This modifies the current matrix """
    #     # Calls __sub__
    #     tempmat = self - mat
    #     self.rows = tempmat.rows[:]
    #     return self

    # def __imul__(self, mat):
    #     """ Add a matrix to this matrix. This modifies the current matrix """
    #     # Possibly not a proper operation
    #     # since this changes the current matrix rank as well.
    #     # Calls __mul__
    #     tempmat = self * mat
    #     self.rows = tempmat.rows[:]
    #     self.rsize, self.csize = tempmat.getRank()
    #     return self

    def __truediv__(self, C):
        if isinstance(C, int) or isinstance(C, float):
            ret = Matrix(self.rsize, self.csize)
            ret.rows = [[i/C for i in row] for row in self.rows]
            return ret
        else:
            raise NotImplementedError


    @classmethod
    def _makeMatrix(cls, rows):
        m = len(rows)
        n = len(rows[0])
        # Validity check
        if any([len(row) != n for row in rows[1:]]):
            raise MatrixError("inconsistent row length")
        mat = Matrix(m, n)
        mat.rows = rows

        return mat

    @classmethod
    def makeRandom(cls, m, n, low=0, high=10):
        """ Make a random matrix with elements in range (low-high) """

        obj = Matrix(m, n)
        for x in range(m):
            obj.rows[x] = [random.randrange(low, high) for i in range(obj.csize)]

        return obj

    @classmethod
    def makeZero(cls, m, n):
        """ Make a zero-matrix of rank (mxn) """

        rows = [[0] * n for x in range(m)]
        return cls.fromList(rows)

    @classmethod
    def makeId(cls, m):
        """ Make identity matrix of rank (mxm) """

        rows = [[0] * m for x in range(m)]
        idx = 0

        for row in rows:
            row[idx] = 1
            idx += 1

        return cls.fromList(rows)

    def copy(self):
        return Matrix.fromList(deepcopy(self.rows))

    @classmethod
    def readStdin(cls):
        """ Read a matrix from standard input """

        print('Enter matrix row by row. Type "q" to quit')
        rows = []
        while True:
            line = sys.stdin.readline().strip()
            if line == 'q': break

            row = [int(x) for x in line.split()]
            rows.append(row)

        return cls._makeMatrix(rows)

    @classmethod
    def readGrid(cls, fname):
        """ Read a matrix from a file """

        rows = []
        for line in open(fname).readlines():
            row = [int(x) for x in line.split()]
            rows.append(row)

        return cls._makeMatrix(rows)

    @classmethod
    def fromList(cls, lst):
        """ Create a matrix by directly passing a list
        of lists """
        listoflists = deepcopy(lst)
        # E.g: Matrix.fromList([[1, 2, 3], [4,5,6], [7,8,9]])
        rows = listoflists[:]
        return cls._makeMatrix(rows)