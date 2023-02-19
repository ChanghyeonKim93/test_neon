#include <iostream>
#include <vector>
#include <random>

#include <arm_neon.h> // NEON datatype

#include "utility/timer.h"

#define DATATYPE float

int main() {
    std::cout << "Welcome!\n";

    std::random_device rd;
    std::mt19937 mersenne(rd()); // Create a mersenne twister, seeded using the random device

    // Create a reusable random number generator that generates uniform numbers between 1 and 6
    std::uniform_int_distribution<> die(1, 2);

    int n_data = 32*1000000;
    std::vector<DATATYPE> a(n_data,0);
    std::vector<DATATYPE> b(n_data,0);

    for(int i = 0; i < n_data; ++i) {
        a[i] = die(mersenne);
        b[i] = die(mersenne);
    }

    std::vector<DATATYPE> r_normal(n_data,0);
    std::vector<DATATYPE> r_neon(n_data,0);

    // Do normally
    timer::tic();
    for(int i = 0; i < n_data; ++i) {
        r_normal[i] = a[i] * b[i];
        r_normal[i] *= a[i];
        r_normal[i] *= b[i];
        r_normal[i] *= a[i];
        r_normal[i] *= b[i];
        r_normal[i] *= b[i];
        r_normal[i] *= b[i];
        r_normal[i] *= b[i];
        r_normal[i] *= b[i];
    }
    std::cout<< "normal time: " << timer::toc(0) << " ms" << std::endl;

    // NEON version
    timer::tic();
    float32x4_t va, vb, vres; // 8 bits x 16 data unsigned int
    DATATYPE* ptr_a = a.data();
    DATATYPE* ptr_b = b.data();
    DATATYPE* ptr_res = r_neon.data();

    int step = 4;
    for(int i = 0; i < n_data/step; ++i)
    {
        va = vld1q_f32(ptr_a);
        vb = vld1q_f32(ptr_b);

        vres = vmulq_f32(va,vb);
        vres = vmulq_f32(vres,va);
        vres = vmulq_f32(vres,vb);
        vres = vmulq_f32(vres,va);
        vres = vmulq_f32(vres,vb);
        vres = vmulq_f32(vres,vb);
        vres = vmulq_f32(vres,vb);
        vres = vmulq_f32(vres,vb);
        vres = vmulq_f32(vres,vb);

        vst1q_f32(ptr_res,vres);

        ptr_a   += step;
        ptr_b   += step;
        ptr_res += step;
    }
    std::cout<< "NEON time: " << timer::toc(0) << " ms" << std::endl;
    // va = vdup_n_u16(4);
    // vb = vdup_n_u16(16);
    
    // Compare results
    int n_different = 0;
    for(int i = 0; i < n_data; ++i){
        if(r_normal[i] != r_neon[i]) ++n_different;
    }
    std::cout << n_different << std::endl;

    return 0;
};