# Proyecto SVM

En esta práctica implementamos una *Suport Vector Machine* para la separación de puntos de dos clases diferentes. Para facilitar el estudio de los métodos de optimización utilizados hemos creado un código **Python** que crea automáticamente un *dataset* y clasifica los puntos a través de SVM.

## Prerequisitos

Para la ejecución del programa asegurarse de tener instalado los paquetes de *numpy*, *amplpy* y *sklearn*:

```
pip install amplpy

pip install numpy

pip install sklearn
```

## Antes de ejecutar

Antes de ejecutar asegurese de indicar el PATH a AMPL en su sistema de archivos. En el *launch.py*, queda indicado un ejemplo de PATH; cambielo manteniendo las comillas simples. 

```
ampl = AMPL(Environment('/home/elias/Escritorio/uni/2º/OM/ampl_linux-intel64'))
```

## Ejecución del código (Entrada)

La ejecución del código requiere a parte de indicar el archivo ejecutable *launch.py* 5 parámetros más:

### PARÁMETROS:
- **option**: nos indica que tipo de problema vamos a resolver <br/>
    >1: primal<br/>
    >2: dual con kernel lineal<br/>
    >3: dual cop kernel gaussiano<br/>
- **num_points**: número de puntos a generar
- **seed**: semilla para la generación aleatoria del dataset
- **nu**: parámetro de la formulación del problema SVM
- **data_type**: indica si generamos los puntos con el algoritmo proporcionado
    por el profesor (gensvmdat), con el algoritmo del swiss roll
    (brazo de gitano) de sklearn o usamos la base de datos "skin"
    >1: Para generarlos con gensvmdat (datos linealmente separables)<br/>
    >2: Para generarlos con sklearn swiss_roll (datos no separables linealmente)<br/>
    >3: Para usar la base de datos "skin"<br/>
  
Si no se introdujese correctamente la entrada, el programa imprimiria por pantalla un aviso recordando el formato de la entrada. Un ejemplo de entrada, donde cada parámetro corresponde en orden a los indicados arriba, seria:

```
python3 launch.py 1 250 1234 0.5 1
```

## Salida

Al ejecutar el programa, se mostrará por pantalla el valor de la función objetivo en el punto óptimo encontrado y se generará un fichero de texto *resultados.txt* donde se guardan los resultados de las variables del problema de optimización así como las *accuracies*, tanto para training como para test. Cabe resaltar que aunque el acierto sobre el conjunto de entrenamiento será siempre igual (los datos se escogen con la *seed* indicada), el acierto sobre los datos de test será diferente en cada ejecución ya que estos se generan o seleccionan con una *seed* aleatoria a través de la función *time*. El resultado para el ejemplo de entrada anterior, indicando que no se desea guardar los *slacks* seria:

```
 + ------------------- RESULTADO DEL PROBLEMA PRIMAL -------------------  +

Resultado de gamma: -4.378653208969142

Resultado de los pesos w:
             [[2.0443755]]
             [[2.23990614]]
             [[2.30687113]]
             [[1.90741462]]

La accuracy de test es: 88.0%.
La accuracy de training es: 89.2%.
```

**Importante**: El *.dat* que se pasa a AMPL se elimina al final del programa. Si se desea mantener el archivo para ver, por ejemplo, el kernel *K* o el formato usado comente la línea 108 del código *launch.py*: ```os.remove('./ampl_data.dat')```.
