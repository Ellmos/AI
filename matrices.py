def PrintMat(mat):
    for row in mat:
        print(row)
    print()


def ToMatrixColumn(mat):
    if type(mat) not in (list, tuple) or type(mat[0]) in (list, tuple):
        raise Exception("ToMatrixColumn: parameter must be one dimensional list or tuple")
    return [[i] for i in mat]

def ToMatrixRow(mat):
    if type(mat) not in (list, tuple) or type(mat[0]) not in  (list, tuple) or len(mat[0]) != 1:
        raise Exception("ToMatrixColumn: parameter must be a matrix column")

    return [i[0] for i in mat]



def AddMat(matA, matB, factor=0, function=()):
    rowA, colA = len(matA), len(matA[0])
    rowB, colB = len(matB), len(matB[0])
    
    if rowA != rowB or colA != colB:
        raise Exception("Add: Matrices have different dimensions")

    if factor and function != ():
        return [[function(x + y * factor) for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]
    if factor:
        return [[(x + y * factor) for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]
    if function != ():
        return [[function(x+y) for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]

    return [[x + y for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]


def SubtractMat(matA, matB, factor=0, function=()):
    rowA, colA = len(matA), len(matA[0])
    rowB, colB = len(matB), len(matB[0])

    if rowA != rowB or colA != colB:
        raise Exception("Subtract: Matrices have different dimensions")

    if factor and function != ():
        return [[function(x - y * factor) for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]
    if factor:
        return [[(x - y * factor) for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]
    if function != ():
        return [[function(x - y) for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]

    return [[x - y for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]

def MultiplyMatBis(matA, matB, factor=0, function=()):
    rowA, colA = len(matA), len(matA[0])
    rowB, colB = len(matB), len(matB[0])

    if rowA != rowB or colA != colB:
        raise Exception("MultiplyBis: Matrices have different dimensions")

    if factor and function != ():
        return [[function(x * y * factor) for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]
    if factor:
        return [[x * y * factor for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]
    if function != ():
        return [[function(x * y) for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]

    return [[x * y for (x, y) in zip(matA[i], matB[i])] for i in range(rowA)]


def MultiplyMatByScalar(mat, x):
    return [[i*x for i in row] for row in mat]

def MultiplyMat(matA, matB):
    rowA, colA = len(matA), len(matA[0])
    rowB, colB = len(matB), len(matB[0])

    if colA != rowB:
        raise Exception("Multiply: Matrices can't be multiply")

    res = []

    for i in range(rowA):
        newRow = []
        for j in range(colB):
            tmp = 0
            for k in range(colA):
                tmp += matA[i][k] * matB[k][j]
            newRow.append(tmp)

        res.append(newRow)

    return res


def TransposeMat(mat):
    rows, cols = len(mat), len(mat[0])

    res = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            res[j][i] = mat[i][j]

    return res
