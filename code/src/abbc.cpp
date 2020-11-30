#include <celerity/celerity.h>
#include <iostream>
#include <vector>
#include <chrono>

void abbc(celerity::distr_queue& queue,
          celerity::buffer<float, 2>& buff_a,
          celerity::buffer<float, 2>& buff_b,
          celerity::buffer<float, 2>& buff_c,
          celerity::buffer<float, 2>& buff_d,
          celerity::buffer<float, 2>& buff_e,
          celerity::buffer<float, 2>& buff_f,
          celerity::buffer<float, 2>& buff_g,
          int size){
  queue.submit([=](celerity::handler& cgh){
      auto A = buff_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto B = buff_b.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto C = buff_c.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<2>());
      cgh.parallel_for<class abbc1>(cl::sycl::range<2>(size,size),[=](cl::sycl::item<2> item){
          for(size_t k=0;k<size;k++)
            C[item] = A[{k,item[1]}] * B[{item[0],k}];
          });
  });

  queue.submit([=](celerity::handler& cgh){
      auto D = buff_d.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto C = buff_c.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto E = buff_e.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<2>());
      cgh.parallel_for<class abbc2>(cl::sycl::range<2>(size,size),[=](cl::sycl::item<2> item){
          for(size_t k=0;k<size;k++)
            E[item] = D[{k,item[1]}] * C[{item[0],k}];
          });
  });
  queue.submit([=](celerity::handler& cgh){
      auto C = buff_c.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
      auto F = buff_f.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
      auto G = buff_g.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<2>());
      cgh.parallel_for<class abbc3>(cl::sycl::range<2>(size,size),[=](cl::sycl::item<2> item){
          for(size_t k=0;k<size;k++)
            G[item] = C[{k,item[1]}] * F[{item[0],k}];
          });
  });
}

int main(int argc, char* argv[]){
  if(argc <= 1){
    std::cerr<<"invalid size"<<std::endl;
    return 1;
  }
  auto size = std::stoi(argv[1]);

  std::vector<float> A(size * size);
  std::vector<float> B(size * size);
  std::vector<float> C(size * size);
  std::vector<float> D(size * size);
  std::vector<float> E(size * size);
  std::vector<float> F(size * size);
  std::vector<float> G(size * size);
  
  srand(42);
  for(int i=0;i<size;i++)
    for(int j=0;j<size;j++){
      A[i*size+j] = rand();
      B[i*size+j] = rand();
      D[i*size+j] = rand();
      F[i*size+j] = rand();
    }

  celerity::buffer<float, 2> buff_A(A.data(), cl::sycl::range<2>(size,size)); 
  celerity::buffer<float, 2> buff_B(B.data(), cl::sycl::range<2>(size,size)); 
  celerity::buffer<float, 2> buff_C(C.data(), cl::sycl::range<2>(size,size)); 
  celerity::buffer<float, 2> buff_D(D.data(), cl::sycl::range<2>(size,size)); 
  celerity::buffer<float, 2> buff_E(E.data(), cl::sycl::range<2>(size,size)); 
  celerity::buffer<float, 2> buff_F(F.data(), cl::sycl::range<2>(size,size)); 
  celerity::buffer<float, 2> buff_G(G.data(), cl::sycl::range<2>(size,size)); 

  celerity::distr_queue queue;
  const auto t1 = std::chrono::high_resolution_clock::now();
  abbc(queue,
       buff_A,
       buff_B,
       buff_C,
       buff_D,
       buff_E,
       buff_F,
       buff_G,
       size);
  queue.slow_full_sync();
  const auto t2 = std::chrono::high_resolution_clock::now();
  
  queue.with_master_access([&](celerity::handler& cgh) {
    cgh.run([&]() {
      const auto duration =std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1); 
			fprintf(stdout, "GPU Runtime: %lf s\n", (duration.count() / 1.0e9));
    });
  });
  return 0;
}
