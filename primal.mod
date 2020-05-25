# Parameters
param n >= 1, integer;
param m >= 1, integer;
param nu >= 0;

param A {1..m, 1..n};
param y {i in 1..m};


# Variables
var gamma;
var w {1..n};
var s {1..m} >= 0;

#optimización
minimize SVM_primal: 1/2*sum{i in 1..n}(w[i]*w[i]) + nu*sum{j in 1..m}(s[j]);
subject to restric {j in 1..m}:
	y[j] * (sum{i in 1..n} (w[i]*A[j,i]) + gamma) + s[j] >= 1;