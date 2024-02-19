//#include <sycl/sycl.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
// #include <getopt.h>

using namespace std;

constexpr size_t Niter = 30000;

template<class T>
void matvec_cpu(vector<size_t> C,vector<size_t> R,vector<T>V,vector<T>u,vector<T> &res) {
    size_t n = R.size()-1;
for (size_t K=0; K<Niter; K++) 
    for(size_t i=0; i<n; i++)
    {
        res[i] = T{};
        for (size_t j=R[i]-1; j<R[i+1]; j++)
        {
            res[i] = res[i] + V[j]*u[C[j]-1];
        }
    }
}

template<class T>
void matvec_gpu(vector<size_t> C,vector<size_t> R,vector<T>V,vector<T>u,vector<T> &res, double &duration)
{
    using namespace Kokkos;
    size_t n = R.size()-1;

    // buffer<size_t,1> b_C(C.data(), range<1>(C.size())), 
                    //  b_R(R.data(), range<1>(R.size()));
    // buffer<T,1> b_V(V.data(), range<1>(V.size()));
    // buffer<T,1> b_u(u.data(), range<1>(u.size()));
    // buffer<T,1> b_res(res.data(), range<1>(res.size()));

    size_t* b_C = (size_t*) kokkos_malloc<SharedSpace>(C.size() * sizeof(size_t));
    size_t* b_R = (size_t*) kokkos_malloc<SharedSpace>(R.size() * sizeof(size_t));
    T* b_V = (T*) kokkos_malloc<SharedSpace>(V.size() * sizeof(T));

    T* b_u = (T*) kokkos_malloc<SharedSpace>(u.size() * sizeof(T));
    T* b_res = (T*) kokkos_malloc<SharedSpace>(res.size() * sizeof(T));

    memcpy(b_C, C.data(), C.size() * sizeof(size_t));
    memcpy(b_R, R.data(), R.size() * sizeof(size_t));
    memcpy(b_V, V.data(), V.size() * sizeof(T));
    memcpy(b_u, u.data(), u.size() * sizeof(T));

    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    for (size_t K=0; K<Niter; K++)  {
        parallel_for(n, KOKKOS_LAMBDA(const int i) {
                b_res[i] = T{};
                for (size_t j=b_R[i]-1; j<b_R[i+1]; j++)
                    b_res[i] += b_V[j]*b_u[b_C[j]-1];
            });
        fence();
    }

    duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
    //memcpy(res.data, b_u, u.size()*sizeof(T));
    for (size_t K=0; K<res.size(); K++)
      res[K] = b_res[K];
    kokkos_free<SharedSpace>(b_C);
    kokkos_free<SharedSpace>(b_R);
    kokkos_free<SharedSpace>(b_V);
    kokkos_free<SharedSpace>(b_u);
    kokkos_free<SharedSpace>(b_res);
    
}

int main(int argc, char* argv[])
{
    vector<size_t> C,R;
    vector<double> V;
    ifstream inputFile; 
    size_t number;
    double f_number;

    inputFile.open("C.csv");
    while(inputFile >> number) C.push_back(number);
    inputFile.close();

    inputFile.open("R.csv");
    while(inputFile >> number) R.push_back(number);
    inputFile.close();

    inputFile.open("V.csv");
    while(inputFile >> f_number) V.push_back(f_number);
    inputFile.close();

    vector<double> u(R.size()-1), v(R.size()-1);
    u[R.size()/2-1] = 10;
    auto kernel_duration=0.0;
    double duration;
    Kokkos::initialize(argc, argv);
    matvec_gpu(C,R,V,u,v, duration);

    cout << "u at " << R.size()/2-1 << " equals " << u[R.size()/2-1] << endl;
    
    cout << "v: " << endl;
    for(size_t k=485; k<500; k++)
      cout << "row: " << k+1 << " value: " << v[k] << endl; 

    std::cout << "Compute Duration      : " << duration / 1e+9 << " seconds\n";
    Kokkos::finalize();

    return 0;
}