import math
import numpy as np
import scipy.linalg as sc
import timeit

def readMatrixInputFromFile(fileName):
    '''
    Return Data read from the text file\n
    :param fileName: file name
    :return: Data read from the text file
    '''

    return np.loadtxt(fileName, dtype=np.float64, delimiter=',')

def calcTylor(matrix):
    # get the size of the matrix
    rowNum, colNum = np.shape(matrix)
    # create a unit matrix depends on the size of the given matrix
    unitMatrix = np.eye(rowNum, colNum)
    # find the C value for Lagrange Reminder
    c = np.sqrt(power_iteration(matrix))

    epsilon = 0.0000000001
    tylorMatrix = unitMatrix + matrix
    multiplyedMatrix = matrix
    multiplyedDenom = 1
    i = 2

    while True:
        multiplyedMatrix = np.dot(multiplyedMatrix, matrix)
        multiplyedDenom = multiplyedDenom * i
        coefficient = 1 / multiplyedDenom
        tylorMatrix = tylorMatrix + (coefficient * multiplyedMatrix)

        #calculate reminder
        coefficient_reminder = np.power(math.e,c) / multiplyedDenom * (i+1)
        reminder = coefficient_reminder * math.pow(c,i+1)
        i = i + 1
        #print(i)

        if reminder < epsilon:
            break

    return tylorMatrix

def calcAtA(matrix):
    # calculate transpose_matrix * matrix
    tranposeMatrix = np.transpose(matrix)
    aTa = tranposeMatrix.dot(matrix)
    return aTa

def calcVectorNorm(vector):
    return np.sqrt(np.dot(vector,vector))

def eigenvalue(A, v):
    '''
    Return the eigenvalue
    Av =xV
    :param A: matrix
    :param v: vector
    :return: eigenvalue
    '''
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration(A):
    '''
    Return the biggest eigenvalue and eigenvector
    The basic idea is to start with a random vector, and then repeatedly multiply it by the matrix to find a vector that is "more aligned" with the largest eigenvector.
    The process is repeated until the vector stops changing much.
    The final vector that you get is the eigenvector associated with the largest eigenvalue.

    How the Algorithm work?
    Start with an initial vector, often chosen randomly.
    Multiply this vector by the matrix.
    Normalize the resulting vector.
    Repeat steps 2 and 3 a certain number of times or until the vector stops changing much.
    :param A: matrix
    :return: the biggest eigenvalue of a matrix
    '''
    row, col = A.shape

    v = np.ones(col) / np.sqrt(col)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / calcVectorNorm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.0000000001:
            break

        ev = ev_new

    return ev_new


#main
input_matrix = readMatrixInputFromFile("inv_matrix(1000x1000).txt")

start = timeit.default_timer()
our_exponential_matrix = calcTylor(input_matrix)
stop = timeit.default_timer()
print('Our Time: ', stop - start)

start = timeit.default_timer()
python_exponential_matrix = sc.expm(input_matrix)
stop = timeit.default_timer()
print('Python Time: ', stop - start)



print('end.')

















