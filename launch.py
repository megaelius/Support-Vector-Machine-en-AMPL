from amplpy import AMPL, Environment
import sys
import functions as fun
import numpy as np
import os


# Path donde está AMPL en nuestro ordenador
ampl = AMPL(Environment('/home/elias/Escritorio/uni/2º/OM/ampl_linux-intel64'))

# Escogemos el solver cplex
ampl.setOption('solver', 'cplex')

'''
PARÁMETROS:
- option: nos indica que tipo de problema vamos a resolver
    1: primal
    2: dual con kernel lineal
    3: dual cop kernel gaussiano
- num_points: número de puntos
- seed: semilla para la generación aleatoria del dataset
- nu: parámetro de la formulación del problema SVM
- swiss: indica si generamos los puntos con el algoritmo del swiss roll
         (brazo de gitano) de sklearn o el algoritmo proporcionado
         por el profesor (gensvmdat)
    0: Para generar los puntos con gensvmdat (separación lineal)
    1: Para generar los puntos con sklearn swiss_roll (separación no lineal)
'''
try:
    option = int(sys.argv[1])
    num_points = int(sys.argv[2])
    seed = int(sys.argv[3])
    nu = float(sys.argv[4])
    swiss = int(sys.argv[5])


    A, y = fun.generate_data(option, num_points, seed, nu, swiss)

    # Leemos el modelo y los datos
    if option == 1: ampl.read('./primal.mod')
    else: ampl.read('./dual.mod')

    ampl.readData('./ampl_data.dat')
    ampl.solve()

    '''
    Aquí obtenemos los parámetros y variables que nos da AMPL y hacemos la
    conversión a matrices de numpy
    '''
    if option == 1:
        w = ampl.getVariable('w').getValues()
        w = np.matrix(w.toPandas())
        w = w.reshape(w.size, 1)

        gamma = ampl.getVariable('gamma').getValues()
        gamma = float(gamma.toList()[0])

        s = ampl.getVariable('s').getValues()
        s = np.matrix(s.toPandas())
        s = s.reshape(num_points, 1)

        c1 = fun.primal_classification(A, w, y, gamma, s)
    else:
        la = ampl.getVariable('lambda').getValues()
        la = np.matrix(la.toPandas())
        la = la.reshape(num_points, 1)

        K = ampl.getParameter('K').getValues()
        K = np.matrix(K.toPandas())
        K = K.reshape(num_points, num_points)
        clasif, gamma = fun.dual_classification(la,y,K,nu)

    '''
    Imprimimos los resultados en el archivo resultados.txt
    '''
    if option == 1:
        fun.print_to_txt(w = w, gamma = gamma, acc1 = fun.precision(y, c1), s = s, option = 1)
    elif option == 2:
        fun.print_to_txt(w = fun.dual_w(la, y, A, nu), gamma = gamma, acc1 = fun.precision(y, clasif), option = 2)
    else:
        fun.print_to_txt(gamma = gamma, acc1 = fun.precision(y, clasif), option = 3)

    os.remove('./ampl_data.dat') # Eliminamos el .dat que le pasamos a AMPL y nos guardamos únicamente
                                 # el de generación. Para mantener el .dat comentar esta línea.

# Error de formato de entrada
except:
    print('ERROR EN LA ENTRADA!: Por favor, asegurese de introducir además ' +
          'del archivo ejecutable (EN ESTE ORDEN) la opción, el número de ' +
          'puntos a generar, la seed para la generación de puntos, la nu ' +
          'y un útlimo parámetro que indique como generar los datos:\n\n' +
          '         0: Para generarlos con gensvmdat (separación lineal)\n' +
          '         1: Para generarlos con sklearn swiss_roll (separación no lineal)\n')
    print('Ejemplo: python3 launch.py 1 250 1234 0.5 0')
    print('\nOpciones disponibles:\n' +
          '         1. Problema primal\n' +
          '         2. Problema dual\n' +
          '         3. Problema dual (RBF)\n')
