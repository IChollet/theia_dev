#ifndef THEIA_MATRICES_LOW_RANK_MATRICES_HPP
#define THEIA_MATRICES_LOW_RANK_MATRICES_HPP

namespace theia{

  template<typename FLT>
  struct lrmat{
    int  m;   // Number of rows before factorization
    int  n;   // Number of colums before factorization
    int  r;   // Rank
    FLT* U;   // Left  side
    FLT* V;   // Right side 
  }; // LRMAT

  template<typename FLT>
  inline void allocate(lrmat<FLT>* UV, int m_, int n_, int r_){
    (*UV).m = m_;
    (*UV).n = n_;
    (*UV).r = r_;
    (*UV).U = (FLT*)std::malloc(sizeof(FLT)*m_*r_);
    (*UV).V = (FLT*)std::malloc(sizeof(FLT)*n_*r_);
  }

} // THEIA

#endif
