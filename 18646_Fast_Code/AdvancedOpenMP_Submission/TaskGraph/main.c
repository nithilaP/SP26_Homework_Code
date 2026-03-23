#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <cmath>

using namespace std;

void test_routine(double* a, int N)
{

  for (int i = 0; i != N; ++i)
    {
      a[i*N+i] = sqrt(a[i*N+i]);

      for (int j = i+1; j <N; ++j)
	{
	  a[j*N+i] = a[j*N+i] / a[i*N+i];
	}

      
      for (int j = i + 1; j < N; ++j)
	{
	  for (int k = i+1; k <= j; ++k)
	    {
	      a[j*N+k] -= a[j*N+i]*a[k*N+i];
	    }
	}
      
    }
 
}



int main()
{

  unsigned long long st, et;

  int N = 10;

  double *a, *b;
  a = (double*)calloc(N*N, sizeof(double));
  b = (double*)calloc(N*N, sizeof(double));  
  
  //create lower tri matrix
  for (int i = 0; i < N; ++i)
    for (int  j = 0; j <= i; ++j)
      a[i*N+j] = rand() %10 + 1;

  //compute A * A'
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
	b[i*N+j] += a[i*N+k] * a[j*N+k];
  
  test_routine(b, N);

  bool correct = true;
  for (int i = 0; i < N; ++i)
    for (int  j = 0; j <= i; ++j)
      correct &= (fabs(a[i*N+j] - b[i*N+j]) < 1e-7);

  cout<<(correct?"Yes":"No")<<endl;
  
  free(a);
  free(b);  
  
  return 0;
}

