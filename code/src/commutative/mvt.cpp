#include <celerity/celerity.h>
#include <iostream>
#include <vector>
#include <chrono>

void mvt(celerity::distr_queue& queue,
          celerity::buffer<float, 2>& buff_a,
          celerity::buffer<float, 2>& buff_x1,
          celerity::buffer<float, 2>& buff_x2,
          celerity::buffer<float, 2>& buff_y1,
          celerity::buffer<float, 2>& buff_y2,
          int size){
    using namespace cl::sycl;
    using namespace celerity::access;
    queue.submit([=](celerity::handler& cgh) {
        auto a = buff_a.template get_access<access::mode::read>(cgh, slice<2>(1));
        auto y1 = buff_y1.template get_access<access::mode::read>(cgh, slice<2>(0));
        auto x1 = buff_y2.template get_access<access::mode::read_write>(cgh, one_to_one<2>());
        cgh.parallel_for<class Mvt1>(buff_x1.get_range(), [=, N_ = size](item<2> item) {
            const auto i = item[0];
            for(size_t j = 0; j < N_; j++) {
                x1[{i, 0}] += a[{i, j}] * y1[{j, 0}];
            }
        });
    });

    queue.submit([=](celerity::handler& cgh) {
        auto a = buff_y2.template get_access<access::mode::read>(cgh, slice<2>(1));
        auto y2 = buff_a.template get_access<access::mode::read>(cgh, slice<2>(0));
        auto x2 = buff_x2.template get_access<access::mode::read_write>(cgh, one_to_one<2>());
        cgh.parallel_for<class Mvt2>(buff_y1.get_range(), [=, N_ = size](item<2> item) {
            const auto k = item[0];
            for(size_t l = 0; l < N_; l++) {
                x2[{k, 0}] += a[{l, k}] * y2[{l, 0}];
            }
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
  std::vector<float> x1(size,0);
  std::vector<float> x2(size,0);
  std::vector<float> y1(size,0);
  std::vector<float> y2(size,0);
  
  srand(42);
  for(int i=0;i<size;i++)
    for(int j=0;j<size;j++){
      A[i * size + j] = (float)(i + j + 1.0) / size;
    }

  celerity::buffer<float, 2> buff_A(A.data(), cl::sycl::range<2>(size,size)); 
  celerity::buffer<float, 2> buff_x1(x1.data(), cl::sycl::range<2>(size,1)); 
  celerity::buffer<float, 2> buff_x2(x2.data(), cl::sycl::range<2>(size,1)); 
  celerity::buffer<float, 2> buff_y1(y1.data(), cl::sycl::range<2>(size,1)); 
  celerity::buffer<float, 2> buff_y2(y2.data(), cl::sycl::range<2>(size,1)); 

  celerity::distr_queue queue;
  const auto t1 = std::chrono::high_resolution_clock::now();
  mvt(queue,
       buff_A,
       buff_x1,
       buff_x2,
       buff_y1,
       buff_y2,
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
