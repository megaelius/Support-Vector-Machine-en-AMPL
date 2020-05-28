from amplpy import AMPL, Environment
import functions as fun
import numpy as np
import random
import time
import sys
import os


# Path donde está AMPL en nuestro ordenador
ampl = AMPL(Environment('/home/alex/AMPL/ampl_linux-intel64'))
#ampl = AMPL(Environment('/home/elias/Escritorio/uni/2º/OM/ampl_linux-intel64'))
# Escogemos el solver cplex
ampl.setOption('solver', 'cplex')

'''
PARÁMETROS:
- option: nos indica que tipo de problema vamos a resolver
    1: primal
    2: dual con kernel lineal
    3: dual con kernel gaussiano
- num_points: número de puntos
- seed: semilla para la generación aleatoria del dataset
- nu: parámetro de la formulación del problema SVM
- data_type: indica si generamos los puntos con el algoritmo proporcionado
    por el profesor (gensvmdat), con el algoritmo del swiss roll
    (brazo de gitano) de sklearn o usamos la base de datos "skin"
    1: Para generarlos con gensvmdat (datos linealmente separables)
    2: Para generarlos con sklearn swiss_roll (datos no separables linealmente)
    3: Para usar la base de datos "skin"
'''
try:
    option = int(sys.argv[1])
    num_points = int(sys.argv[2])
    seed = int(sys.argv[3])
    nu = float(sys.argv[4])
    data_type = int(sys.argv[5])

    '''
    Aquí generamos los dos datasets:
        El de training, para generar el modelo, es decir, w y gamma.

        El de test, para clasificar nuevos puntos con el modelo anterior
        y ver cómo de bien clasifica puntos con los que no ha modelado.
    '''

    print("\nCalculando resultados...\n")

    Atr, ytr = fun.generate_data(num_points, seed, data_type, False)
    fun.write_ampl(Atr, ytr, nu, option)

    if option != 3:
        if data_type != 3:
            random.seed(time.time())
            seed2 = random.randint(0, 1e6)
        else: seed2 = seed
        Ate, yte = fun.generate_data(num_points, seed2, data_type, True)

    # Leemos el modelo y los datos
    if option == 1: ampl.read('./primal.mod')
    else: ampl.read('./dual.mod')

    ampl.readData('./ampl_data.dat')
    ampl.solve()

    print("\nEscribiendo resultados...\n")

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

        ctrain = fun.primal_classification(Atr, w, ytr, gamma)

    else:
        la = ampl.getVariable('lambda').getValues()
        la = np.matrix(la.toPandas())
        la = la.reshape(num_points, 1)

        K = ampl.getParameter('K').getValues()
        K = np.matrix(K.toPandas())
        K = K.reshape(num_points, num_points)

        ctrain, gamma = fun.dual_classification(la, ytr, K, nu)
        if option == 2:
            w = fun.dual_w(la, ytr, Atr, nu)

    if option != 3:
        ctest = fun.primal_classification(Ate, w, yte, gamma)

    '''
    Imprimimos los resultados en el archivo resultados.txt
    '''
    if option == 1 or option == 2:
        fun.print_to_txt(w = w, gamma = gamma, acc1 = fun.precision(ytr, ctrain), acc2 = fun.precision(yte, ctest), option = option)
    else:
        fun.print_to_txt(gamma = gamma, acc1 = fun.precision(ytr, ctrain), option = 3)

    os.remove('./ampl_data.dat')
    # Eliminamos el .dat que le pasamos a AMPL y nos guardamos únicamente
    # el de generación. Para mantener el .dat comentar esta línea.

# Error de formato de entrada
except:
    print('\nERROR EN LA ENTRADA!: Por favor, asegurese de introducir además ' +
          'del archivo ejecutable (EN ESTE ORDEN) la opción, el número de ' +
          'puntos a generar, la seed para la generación de puntos, la nu ' +
          'y un útlimo parámetro que indique como generar los datos:\n\n' +
          '         1: Para generarlos con gensvmdat (datos linealmente separables)\n' +
          '         2: Para generarlos con sklearn swiss_roll (datos no separables linealmente)\n' +
          '         3: Para usar la base de datos "skin"\n')
    print('Ejemplo: python3 launch.py 1 250 1234 0.5 1')
    print('\nOpciones disponibles:\n' +
          '         1. Problema primal\n' +
          '         2. Problema dual\n' +
          '         3. Problema dual (RBF)\n')
