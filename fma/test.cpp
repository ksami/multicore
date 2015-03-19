#include <emmintrin.h>

#define N 1000

// ONLY tested timing with integers
// TODO: run tests on floats to compare accuracy
int main(void) {
	__m64 a;
	__m64 b;
	__m64 c;
	__m64 res;

	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			for(int k=0; k<N; k++) {
				a = _mm_set_pi32(0, i);
				b = _mm_set_pi32(0, j);
				c = _mm_set_pi32(0, k);
				res = _mm_madd_pi16(a, b);
				//res = _mm_add_pi32(_mm_mul_su32(a, b), c);
			}
		}
	}

	return 0;
}
