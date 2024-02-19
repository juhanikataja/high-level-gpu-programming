#include <sycl/sycl.hpp>
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
    using namespace sycl;
    queue q;
    size_t n = R.size()-1;
    buffer<size_t,1> b_C(C.data(), range<1>(C.size())), b_R(R.data(), range<1>(R.size()));
    buffer<T,1> b_V(V.data(), range<1>(V.size()));
    buffer<T,1> b_u(u.data(), range<1>(u.size()));
    buffer<T,1> b_res(res.data(), range<1>(res.size()));

    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
for (size_t K=0; K<Niter; K++)  {
    q.submit([&](handler &h) {
        accessor a_C{b_C, h};
        accessor a_R{b_R, h};
        accessor a_V{b_V, h};
        accessor a_u{b_u, h};
        accessor a_res{b_res, h};
        h.parallel_for(range{n}, [=](const size_t i) {
            a_res[i] = T{};
            for (size_t j=a_R[i]-1; j<a_R[i+1]; j++)
              a_res[i] += a_V[j]*a_u[a_C[j]-1];
        });
    }).wait();
}
    duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
    
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
    matvec_gpu(C,R,V,u,v, duration);

    cout << "u at " << R.size()/2-1 << " equals " << u[R.size()/2-1] << endl;
    
    cout << "v: " << endl;
    for(size_t k=485; k<500; k++)
      cout << "row: " << k+1 << " value: " << v[k] << endl; 

    std::cout << "Compute Duration      : " << duration / 1e+9 << " seconds\n";

    return 0;
}