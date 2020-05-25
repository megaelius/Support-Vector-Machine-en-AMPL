import numpy as np
import os


#If the .py is saved in the same folder of gensvmdat, change path to "".
#Otherwise put the correct folder path and the new files will be stored there.
path = "/home/victor/Escritorio/Matlab/OM/Proyecto2"

#Nombre de punts a generar.
num_points = 250

seed = 1234

nu_value = 0.5

os.system(path+"/gensvmdat data_D.dat "+str(num_points)+" "+str(seed))

with open(path+"/data_D.dat","r") as raw, \
     open(path+"/clean_data_D.dat","w") as clean:
        data = raw.read()
        data = data.replace('*','').replace('   ',' ')
        clean.write(data)

#If you want to keep the file with asterisks, comment this line.
os.remove(path+"/data_D.dat")

#If you also want to remove the clean data, uncomment this line.
#os.remove(path+"/clean_data_D.dat")


A = np.loadtxt(path+"/clean_data_D.dat", delimiter=' ')
y = A[:,A[0].size-1]
A = np.delete(A,A[0].size-1,1)


K = np.dot(A,A.T)
K = np.matrix(K)
aux = K[:,0].size


for i in range(1,3):
    o = open(path+"/input_D.dat","w")
    o.write("param m := "+str(aux)+";\n")
    o.write("param nu:= "+str(nu_value)+";\n\n")
    o.write("param K :\n    ")
    for i in range (1,aux+1):
        o.write(str(i)+"     ")
    o.write(":=\n")
    for i in range(0,aux):
        o.write(str(i+1)+" ")
        o.write(str(K[i,:]).replace('[','').replace(']','')+"\n")
    o.write(";\n\nparam y :=\n")
    for i in range(0,aux):
        o.write(str(i+1) + " " + str(y[i]) +"\n")
    o.write(";")


print("Finished!")
