//#include <emmintrin.h>
#include <iostream>
#include <limits>
#include <x86intrin.h>

#define N 100000
#define A 100.1
#define B 0.03

using namespace std;

typedef numeric_limits<double> dbl;

int main(void) {
  __m128d a;
  __m128d b;
  __m128d res;
  double result;

  double arrA[N];
  double arrB[N];

  //array init
  for(int i=0; i<N; i++) {
    arrA[i] = 5e-3;
    arrB[i] = 3e-6;
  }

  res = _mm_setzero_pd();  //sets both upper and lower to 0.0

  for(int i=0; i<N; i++) {
    a = _mm_set_sd(arrA[i]);  //sets lower 64 to num, upper to 0.0
    for(int j=0; j<N; j++) {
      b = _mm_set_sd(arrB[j]);
      res = _mm_add_sd(_mm_mul_sd(a, b), res);
      //res = _mm_fmadd_sd(a, b, res);
    }
  }

  result = _mm_cvtsd_f64(res);

  cout.precision(dbl::digits10);
  cout << "result is " << fixed << result << endl;

  return 0;
}
