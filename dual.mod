# Parameters
param n >= 1, integer;
param m >= 1, integer;
param nu >= 0;

param K {1..m, 1..m};
param y {i in 1..m};

# Variables
var gamma;
var lambda {1..m} >= 0, <= nu;

#optimización
maximize SVM_dual: sum{i in 1..m}(lambda[i]) 
                   -1/2*sum{j in 1..m, k in 1..m}( 
                       lambda[j]*y[j]*lambda[k]*y[k]*K[j,k]
                   );
subject to Dual_restric: sum{i in 1..m}(lambda[i]*y[i]) = 0;