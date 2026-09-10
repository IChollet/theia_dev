# Cubature tutorial

### Tensor rules
A quadrature rule (i.e. 1D rule), for instance a Gauss-Hermite one with order 5, can be obtained in the following way
```c
#include "***/theia_dev/cubature/cubature.hpp"
auto rule = theia::cubature::gauss_hermite::get<FLT>(5);
```
FLT being float / double.

To get the nodes and weigths values printed on terminal, one can then simply call
```c
print(rule);
```

All cubature rules share the same structre type (with different templates), in which you can access to nodes, weights and size. Here is the interface (as SoA).
```c
int  size    = rule.N; // Total number of points
auto nodes   = rule.x; // List of nodes (std::vector<std::array<NODE_t,DIM>>)
auto weights = rule.w; // List of weights (std::vector<WGHT_t>)
```

It is possible to tensorise any cubature rule with any other one using the *tensor_rule* method. Here is an example
```c
auto rule2 = theia::tensor_rule(rule,rule);
```
that tensorises *rule* with itself, creating a 2D tensor rule.

Given a functor ' F ', any rule can be evaluated with a result stored in 'result' using
```c
eval(rule,F,result);
```

### List of 1D rules
- gauss_hermite
- gauss_laguerre
- gauss_legendre

### Cauchy integral
Cauchy contour integral also are available with a slightly different prototype. Please check "test_cauchy.cpp" to get application example.