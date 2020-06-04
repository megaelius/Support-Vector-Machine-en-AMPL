from sklearn import datasets as ds
from os import getcwd
import numpy as np
import random
import time
import os

# Epsilon es la tolerancia de error a la hora de comparar números
eps = 10e-6


'''
Recibe como parámetros el número de puntos y las variables y parámetros
y las escribe en un archivo en el formato legible para AMPL.
'''
def write_data(nu, K, y, laux):
        dimensions = len(K[0])

        o = open('./ampl_data.dat', 'w')
        if (laux == 'A'): o.write('param n := ' + str(dimensions) + ';\n')
        o.write('param m := ' + str(len(y)) + ';\n' + \
                'param nu:= ' + str(nu) + ';\n\n' + \
                'param ' + laux + ':\n    ')
        for i in range (1, dimensions + 1):
            o.write(str(i) + '     ')
        o.write(':=\n')
        for i in range(0, len(y)):
            o.write(str(i + 1) + ' ' + \
                    str(K[i, :]).replace('[', '').replace(']', '') + '\n')
        o.write(';\n\nparam y :=\n')
        for i in range(0, len(y)):
            o.write(str(i + 1) + ' ' + str(y[i]) +'\n')
        o.write(';')


'''
Se generan num_points puntos con su clase aleatoriamente mediante el método
gensvmdat.
'''
def gensvmdat(num_points, seed):
    os.system('./gensvmdat data.dat ' + str(num_points) + ' ' + str(seed))
    with open('./data.dat', 'r') as raw, \
         open('./aux_data.dat', 'w') as clean:
            data = raw.read()
            data = data.replace('*', '').replace('   ', ' ')
            clean.write(data)

    A = np.loadtxt('./aux_data.dat', delimiter = ' ')
    y = A[:, A[0].size - 1]
    A = np.delete(A, A[0].size - 1, 1)
    return A, y


'''
Se generan num_points puntos con su clase aleatoriamente mediante el método
del swiss_roll (brazo de gitano).
'''
def generate_swiss(num_points,seed):
    A, y = ds.make_swiss_roll(num_points, 2, seed)
    my = np.mean(y)
    y_binary = [0 for i in range(len(y))]
    for i in range(len(y)):
        if y[i] > my: y_binary[i] = 1
        else: y_binary[i] = -1
    return A, y_binary


'''
Debido a errores numéricos, ampl devuelve que la matriz K no es semidefinida
positiva, pero en realidad los valores propios supuestamente negativos, son 0.
Así que sumamos una matriz identidad multiplicada por epsilon para
corregir esto y hacer que la matriz sea semidefinida positiva.
'''
def write_ampl(A, y, nu, option):
    if option == 1:   write_data(nu, A, y, 'A')
    elif option == 2: write_data(nu, A.dot(A.T) + np.eye(len(y))*eps, y, 'K')
    else:
        m = len(y)
        K = np.zeros((m, m))
        s2 = np.mean(np.var(A, 0))
        for i in range(m):
            for j in range(i,m):
                K[i,j] = K[j,i] = np.exp(- np.linalg.norm(A[i,:] - A[j,:])**2/(2*s2))
        write_data(nu, K + np.eye(len(y))*eps, y, 'K')


'''
Con la función generate_skin, leemeos los datos del fichero de texto y los
ordenamos aleatoriamente. Para el conjunto de training cogemos los num_points
primeras observaciones mientras que para los datos de test cogemos aleatoriamente
a través un batch de num_points datos diferentes a los primeros.
'''
def generate_skin(num_points, seed, test):
    A = np.loadtxt('./Skin_NonSkin.txt', delimiter = '	')
    np.random.seed(seed)
    np.random.shuffle(A)
    if not test:
        y = A[:num_points, 3]
        A = A[:num_points, 0:3]
    else:
        random.seed(time.time())
        start = random.randint(num_points, len(A) - num_points - 1)
        y = A[start:(start + num_points), 3]
        A = A[start:(start + num_points), 0:3]

    for i in range(len(y)): y[i] = 2*y[i] - 3  # Transformamos la variable respuesta
    return A, y                                # de 1 / 2 a -1 / 1.


'''
Esta función discierne que tipo de datos quiere generar el usuario y
devuelve la matriz de datos explicativos y un vector con la variable
respuesta.
'''
def generate_data(num_points, seed, dt, test):
    if dt == 1:
        A, y = gensvmdat(num_points, seed)
    elif dt == 2:
        A, y = generate_swiss(num_points, seed)
    else:
        A, y = generate_skin(num_points, seed, test)
    return A, y


'''
A partir de la solución para las lambdas(la), las y's, los puntos y la nu,
devuelve la solución de la w para el caso que utilizamos la formulación dual.
'''
def dual_w(la, y, A, nu):
    w = np.zeros((1,len(A[0,:])))
    for i in range(len(y)):
        if la[i] > eps and la[i] < nu - eps:
            w = w + la[i]*y[i]*A[i,:]
        elif la[i] > nu - eps:
            w = w + nu*y[i]*A[i,:].T
    return w.T


'''
Busca el índice del primer support vector que encuentre y lo devuelve.
'''
def support_vector(la,nu):
    for i in range(len(la)):
        if la[i] > eps and la[i] < nu - eps:
            return i
    return 0


'''
A partir de los resultados de la optimización con la formulación dual
se calculan el parámetro de intersección (gamma) y la clasificación
de cada punto (1 o -1).
'''
def dual_classification(la, y, Atr, nu, option, Ate):
    if option == 3: s2 = np.mean(np.var(Atr, 0))
    m = len(y)
    c = [1 for i in range(m)]
    index_sv = support_vector(la, nu)
    #Calculo gamma
    gamma = 1/y[index_sv]
    for i in range(m):
        if option == 3: K = np.exp(- np.linalg.norm(Atr[index_sv, :] - Atr[i, :])**2/(2*s2))
        else: K = Atr[index_sv, :].dot(Atr[i, :])
        gamma = gamma - la[i]*y[i]*K
    # Miramos en que lado del hiperplano se encuentra el punto.
    for i in range(m):
        wtphi = 0
        for j in range(m):
            if option == 3: K = np.exp(- np.linalg.norm(Ate[i,:] - Atr[j,:])**2/(2*s2))
            else: K = Ate[i, :].dot(Atr[j, :])
            wtphi = wtphi + la[j]*y[j]*K

        if wtphi + gamma < eps: c[i] = -1

    return c, gamma


'''
A partir de los resultados de la optimización con la formulación primal,
calculamos en que lado del hiperplano está cada punto y le asignamos 1 o -1.
'''
def primal_classification(A, w, y, gamma):
    c1 = [1 for i in range(len(y))]
    for i in range(len(y)):
        if A[i,:] * w + gamma < eps: c1[i] = -1

    return c1


'''
A partir de la clase real y la predicha de cada punto, calculamos la precisión
de nuestra optimización.
'''
def precision(y, y_pred):
    acc = 0;
    t = len(y)
    for i in range(t):
        if y[i] == y_pred[i]: acc = acc + 1
    return acc/t


'''
Escribe los resultados en un archivo de texto.
'''
def print_to_txt(w = [0], gamma = 0, acc1 = 0, acc2 = 0, s = 0, option = 1):
    res = open('resultados.txt', 'w')
    if option == 1: res.write('\n + ------------------- RESULTADO DEL PROBLEMA PRIMAL -------------------  +' + '\n\n')
    elif option == 2: res.write('\n + -------------------- RESULTADO DEL PROBLEMA DUAL --------------------  +' + '\n\n')
    else: res.write('\n + -------------------- RESULTADO DEL PROBLEMA RBF --------------------  +' + '\n\n')
    res.write('Resultado de gamma: ' + str(gamma) + '\n')
    if option != 3:
        res.write('\nResultado de los pesos w:' + '\n')
        for i in range(len(w)):
            res.write('             ' + str(w[i]) + '\n')
    res.write('\nLa accuracy de test es: ' + str(acc2*100) + '%.')
    res.write('\nLa accuracy de training es: ' + str(acc1*100) + '%.' + '\n\n')
