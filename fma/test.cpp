//#include <emmintrin.h>
#include <iostream>
#include <limits>
#include <x86intrin.h>

#define N 2000000000
#define A 100.1
#define B 0.03

using namespace std;

typedef numeric_limits<double> dbl;

//extern __m128d _mm_fmadd_sd(__m128d a, __m128d b, __m128d c);

// ONLY tested timing with integers
// TODO: run tests on floats to compare accuracy
int main(void) {
	__m128d a;
	__m128d b;
	__m128d c;
	__m128d res;
	double result;

  res = _mm_setzero_pd();  //sets both upper and lower to 0.0
	
  for(int i=0; i<N; i++) {
		a = _mm_set_sd(A);  //sets lower 64 to num, upper to 0.0
		b = _mm_set_sd(B);
		c = _mm_set_sd(_mm_cvtsd_f64(res));
		// res = _mm_add_sd(_mm_mul_sd(a, b), c);
		res = _mm_fmadd_sd(a, b, c);

		// a = _mm_set_pi32(0, i);
		// b = _mm_set_pi32(0, j);
		// c = _mm_set_pi32(0, k);
		// res = _mm_madd_pi16(a, b);
		// //res = _mm_add_pi32(_mm_mul_su32(a, b), c);
	}

	result = _mm_cvtsd_f64(res);

	cout.precision(dbl::digits10);
	cout << "result is " << fixed << result << endl;

	return 0;
}
