#include <iostream>
#include <limits>

using namespace std;

typedef numeric_limits<double> dbl;

int main(void) {
  long big = 1e9;
  double small = 1e-9;
  double sum = 0.0;
  double ans = 1e9 + 1;


  // Increasing order //
  for(long i=0; i<big; i++) {
    sum += small;
  }
  sum += big;

  // display to the full precision
  cout.precision(dbl::digits10);
  cout << "(increasing) sum is " << fixed << sum
   << ", difference from answer is " << ans-sum << endl;


  sum = 0.0;

  // Decreasing order //
  sum += big;

  for(long i=0; i<big; i++) {
    sum += small;
  }

  // display to the full precision
  cout.precision(dbl::digits10);
  cout << "(decreasing) sum is " << fixed << sum
   << ", difference from answer is " << ans-sum << endl;

  return 0;
}
