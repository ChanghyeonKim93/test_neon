#include <iostream>
#include <vector>
#include <random>

#include <arm_neon.h> // NEON datatype

#include "utility/timer.h"

#define DATATYPE uint16_t

int main() {
    std::cout << "Welcome!\n";

    std::random_device rd;
    std::mt19937 mersenne(rd()); // Create a mersenne twister, seeded using the random device

    // Create a reusable random number generator that generates uniform numbers between 1 and 6
    std::uniform_int_distribution<> die(1, 3);

    int n_data = 32*1000000;

    std::vector<std::vector<DATATYPE>> a(8,std::vector<DATATYPE>(n_data,0));

    register int i;
    for(i = 0; i < n_data; ++i) {
        a[0][i] = die(mersenne);
        a[1][i] = die(mersenne);
        a[2][i] = die(mersenne);
        a[3][i] = die(mersenne);
        a[4][i] = die(mersenne);
        a[5][i] = die(mersenne);
        a[6][i] = die(mersenne);
        a[7][i] = die(mersenne);
    }

    std::vector<DATATYPE> r_normal(n_data,0);
    std::vector<DATATYPE> r_neon(n_data,0);

    // Do normally
    timer::tic(); 
    for(i = 0; i < n_data; ++i) {
        r_normal[i] = a[0][i]
                    + a[1][i]
                    + a[2][i]
                    + a[3][i]
                    + a[4][i]
                    + a[5][i]
                    + a[6][i]
                    + a[7][i];
    }
    float time_normal = timer::toc(0);
    std::cout<< "normal time: " << time_normal << " ms" << std::endl;

    // NEON version
    timer::tic();
    uint16x8_t va[8], vres; // 8 bits x 16 data unsigned int

    DATATYPE* ptr_a[8];
    ptr_a[0] = a[0].data();
    ptr_a[1] = a[1].data();
    ptr_a[2] = a[2].data();
    ptr_a[3] = a[3].data();
    ptr_a[4] = a[4].data();
    ptr_a[5] = a[5].data();
    ptr_a[6] = a[6].data();
    ptr_a[7] = a[7].data();
    DATATYPE* ptr_res = r_neon.data();
    int step = 8;
    for(i = 0; i < n_data/step; ++i)
    {
        va[0] = vld1q_u16(ptr_a[0]); ptr_a[0] += step;
        va[1] = vld1q_u16(ptr_a[1]); ptr_a[1] += step;
        va[2] = vld1q_u16(ptr_a[2]); ptr_a[2] += step;
        va[3] = vld1q_u16(ptr_a[3]); ptr_a[3] += step;
        va[4] = vld1q_u16(ptr_a[4]); ptr_a[4] += step;
        va[5] = vld1q_u16(ptr_a[5]); ptr_a[5] += step;
        va[6] = vld1q_u16(ptr_a[6]); ptr_a[6] += step;
        va[7] = vld1q_u16(ptr_a[7]); ptr_a[7] += step;

        vres = vmovq_n_u16(0.0f);

        vres = vaddq_u16(vres,va[0]);
        vres = vaddq_u16(vres,va[1]); 
        vres = vaddq_u16(vres,va[2]);  
        vres = vaddq_u16(vres,va[3]);  
        vres = vaddq_u16(vres,va[4]);  
        vres = vaddq_u16(vres,va[5]);  
        vres = vaddq_u16(vres,va[6]);  
        vres = vaddq_u16(vres,va[7]);

        vst1q_u16(ptr_res, vres);      
        ptr_res += step;
    }
    float time_neon = timer::toc(0);
    std::cout<< "NEON time: " << time_neon << " ms" << std::endl;

    std::cout << "NEON is faster than normal by : " << time_normal/time_neon << " times.\n";
    // va = vdup_n_u16(4);
    // vb = vdup_n_u16(16);
    
    // Compare results
    float FEPS = 0.001f;
    int n_different = 0;
    for(i = 0; i < n_data; ++i){
        if(abs(r_normal[i] - r_neon[i]) > FEPS) ++n_different;
        // std::cout << "r normal, r_neon: " << (int)(r_normal[i]) <<"," << (int)r_neon[i] << std::endl;
    }
    std::cout << n_different << std::endl;

    return 0;
};