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
auto rule2 = theia::cubature::tensor_rule(rule,rule);
```
that tensorises *rule* with itself, creating a 2D tensor rule.

Rules with different types can be tensorised: the resulting nodes and weights use the promoted types (real x real gives the common real type, and as soon as one of them is complex the result is complex with the common precision). For instance
```c
auto gl = theia::cubature::gauss_legendre::get<double>(12);                 // real nodes / weights
auto ca = theia::cubature::cauchy::get<double>(32, center, radius, point);  // complex nodes / weights
auto t  = theia::cubature::tensor_rule(gl,ca);   // rule<double,2,std::complex<double>,std::complex<double>>
```
The integrand then takes a `std::array<std::complex<double>,2>` (real coordinates have a zero imaginary part).

Given a functor ' F ', any rule can be evaluated with a result stored in 'result' using
```c
eval(rule,F,result);
```

### List of 1D rules
- gauss_hermite
- gauss_laguerre
- gauss_legendre

### Cauchy integral
The rule `cauchy::get<FLT>(N, center, radius)` discretises the circle of given center and radius (trapezoidal rule). It is used with `cauchy::eval` which includes the factor 1/(z - point) in every dimension (see "test_cauchy_double.cpp").

The rule `cauchy::get<FLT>(N, center, radius, point)` includes the factor 1/(z_i - point) in the weights: the generic `eval` then gives (1/2iπ)∮ f(z)/(z-point) dz, i.e. f(point) if the point lies inside the circle (0 outside). Being a standard rule, it can be tensorised with any other rule, for instance one Cauchy rule per dimension with its own point, or a Cauchy rule with a real rule (see "test_cauchy_point_double.cpp" and "test_tensor_rule_mixed_double.cpp").