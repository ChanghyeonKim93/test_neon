#include <iostream>
#include <vector>
#include <random>

#include <arm_neon.h> // NEON datatype

#include "utility/timer.h"

int main() {
    std::cout << "Welcome!\n";

    std::random_device rd;
    std::mt19937 mersenne(rd()); // Create a mersenne twister, seeded using the random device

    // Create a reusable random number generator that generates uniform numbers between 1 and 6
    std::uniform_int_distribution<> die(1, 2);

    int n_data = 16*10000000;
    std::vector<unsigned char> a(n_data,0);
    std::vector<unsigned char> b(n_data,0);

    for(int i = 0; i < n_data; ++i) {
        a[i] = die(mersenne);
        b[i] = die(mersenne);
    }

    std::vector<unsigned char> r_normal(n_data,0);
    std::vector<unsigned char> r_neon(n_data,0);

    // Do normally
    timer::tic();
    for(int i = 0; i < n_data; ++i) {
        r_normal[i] = a[i] * b[i];
        r_normal[i] *= a[i];
        r_normal[i] *= b[i];
        r_normal[i] *= a[i];
        r_normal[i] *= b[i];
    }
    std::cout<< "normal time: " << timer::toc(0) << " ms" << std::endl;

    // NEON version
    timer::tic();
    uint8x16_t va, vb, vres; // 8 bits x 16 data unsigned int
    unsigned char* ptr_a = a.data();
    unsigned char* ptr_b = b.data();
    unsigned char* ptr_res = r_neon.data();
    for(int i = 0; i < n_data/16; ++i)
    {
        va = vld1q_u8(ptr_a);
        vb = vld1q_u8(ptr_b);

        vres = vmulq_u8(va,vb);
        vres = vmulq_u8(vres,va);
        vres = vmulq_u8(vres,vb);
        vres = vmulq_u8(vres,va);
        vres = vmulq_u8(vres,vb);

        vst1q_u8(ptr_res,vres);

        ptr_a   += 16;
        ptr_b   += 16;
        ptr_res += 16;
    }
    std::cout<< "NEON time: " << timer::toc(0) << " ms" << std::endl;
    // va = vdup_n_u16(4);
    // vb = vdup_n_u16(16);
    
    // Compare results
    int n_different = 0;
    for(int i = 0; i < n_data; ++i){
        if(r_normal[i] != r_neon[i]) n_different++;
    }
    std::cout << n_different << std::endl;

    return 0;
};