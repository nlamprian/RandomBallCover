/*! \file 2d_plot.cpp
 *  \brief An application examining the resulting arrangement of the 
 *         `Random Ball Cover` data structure on 2-D data.
 *  \note The application requires further processing to extract 
 *        the representative radii. This feature is not offered by 
 *        the `RBCConstruct` class and is implemented on the CPU.
 *  \author Nick Lamprianidis
 *  \version 1.1
 *  \date 2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2015 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <chrono>
#include <random>
#include <cmath>
#include <CLUtils.hpp>
#include <RBC/algorithms.hpp>
#include <plstream.h>


// Kernel filenames
const std::vector<std::string> kernel_files = { "kernels/rbc_kernels.cl", 
                                                "kernels/scan_kernels.cl" };

// Random generator
auto seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
std::default_random_engine gen (seed);
std::uniform_int_distribution<unsigned int> distc (1, 15);
std::function<unsigned int ()> randColor = std::bind (distc, gen);


/*! \brief Picks a number of elements uniformly at random without replacement from the input sequences.
 *  \details The function implements a Fisher-Yates shuffle.
 *  
 *  \param[in] xBegin input iterator to the initial position of the database 
 *                    sequence with the x dimension.
 *  \param[in] yBegin input iterator to the initial position of the database 
 *                    sequence with the y dimension.
 *  \param[out] rxBegin output iterator to the initial position of the representatives 
 *                      sequence with the x dimension.
 *  \param[out] ryBegin output iterator to the initial position of the representatives 
 *                      sequence with the y dimension.
 *  \param[in] n number of database points.
 *  \param[in] rn number of representative points.
 */
template<typename iterator>
void random_unique (iterator xBegin, iterator yBegin, iterator rxBegin, iterator ryBegin, size_t n, size_t rn)
{
    std::uniform_int_distribution<unsigned long long> dist (0, (unsigned long long) -1);
    std::function<unsigned long long ()> randNum = std::bind (dist, gen);

    size_t left = n;
    while (rn--)
    {
        iterator ix = xBegin;
        iterator iy = yBegin;
        size_t i = randNum () % left;
        std::advance (ix, i);
        std::advance (iy, i);
        std::swap (*xBegin, *ix);
        std::swap (*yBegin, *iy);
        *rxBegin = *xBegin;
        *ryBegin = *yBegin;
        ++xBegin;
        ++yBegin;
        ++rxBegin;
        ++ryBegin;
        --left;
    }
}


class plot
{
public:
    plot (int argc, const char **argv)
    {
        // PLplot parameters
        const int NSIZE = 1024;
        const int RNSIZE = 64;
        const int PADDING = 100;
        std::vector<PLFLT> x (NSIZE), y (NSIZE);
        std::vector<PLFLT> rx (RNSIZE), ry (RNSIZE);
        PLFLT xmin = 0.0, xmax = 1300.0, ymin = 0.0, ymax = 900.0;
        PLFLT _2_PI = 2 * M_PI;

        // OpenCL parameters
        const unsigned int nx = NSIZE;
        const unsigned int nr = RNSIZE;
        const unsigned int d = 4;
        std::vector<PLFLT> radius (RNSIZE, 0);

        pls = new plstream ();

        // Parse command line options
        pls->parseopts (&argc, argv, PL_PARSE_FULL);

        // Set plotting device
        pls->sdev ("xwin");
        // 
        // pls->sdev ("pngqt");
        // pls->setopt ("o", "rbc.png");
        // 
        // pls->sdev ("svgqt");
        // pls->setopt ("o", "rbc.svg");

        // Set color palette
        pls->spal0 ("cmap0_white_bg.pal");

        // Initialize plplot
        pls->init ();

        // Create a labelled box to hold the plot.
        pls->col0 (7);
        pls->env (xmin, xmax, ymin, ymax, 1, -1);
        pls->col0 (15);
        pls->lab ("x#d1#u", "x#d2#u", "Random Ball Cover");
        pls->seed (std::chrono::system_clock::now ().time_since_epoch ().count ());

        // Prepare data
        std::generate (x.begin (), x.end (), 
            [&]() { return (xmax - 2 * PADDING) * pls->randd () + PADDING; } );
        std::generate (y.begin (), y.end (), 
            [&]() { return (ymax - 2 * PADDING) * pls->randd () + PADDING; } );

        // Pick representatives
        random_unique (x.begin (), y.begin (), rx.begin (), ry.begin (), NSIZE, RNSIZE);

        // Setup OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::RBC::KernelTypeC K = cl_algo::RBC::KernelTypeC::SHARED_NONE;
        const cl_algo::RBC::RBCPermuteConfig P = cl_algo::RBC::RBCPermuteConfig::GENERIC;
        cl_algo::RBC::RBCConstruct<K, P> rbc (clEnv, info);
        rbc.init (nx, nr, d);

        // Prepare data for RBCConstruct
        unsigned int idx = 0;
        std::generate ((cl_float4 *) rbc.hPtrInX, (cl_float4 *) rbc.hPtrInX + nx, 
            [&]()
            {
                cl_float p[4];
                p[0] = x[idx]; p[1] = y[idx++];
                p[2] = 0.f; p[3] = 0.f;
                return *((cl_float4 *) p);
            }
        );
        idx = 0;
        std::generate ((cl_float4 *) rbc.hPtrInR, (cl_float4 *) rbc.hPtrInR + nr, 
            [&]()
            {
                cl_float p[4];
                p[0] = rx[idx]; p[1] = ry[idx++];
                p[2] = 0.f; p[3] = 0.f;
                return *((cl_float4 *) p);
            }
        );

        // Copy data to device
        rbc.write (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_IN_X);
        rbc.write (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_IN_R);
        
        rbc.run ();  // Execute kernels

        // Copy results to host
        rbc_dist_id *ID = (rbc_dist_id *) rbc.read (cl_algo::RBC::RBCConstruct<K, P>::Memory::H_OUT_ID, CL_FALSE);
        cl_uint *N = (cl_uint *) rbc.read (cl_algo::RBC::RBCConstruct<K, P>::Memory::H_OUT_N, CL_FALSE);
        cl_uint *O = (cl_uint *) rbc.read (cl_algo::RBC::RBCConstruct<K, P>::Memory::H_OUT_O, CL_FALSE);
        cl_float *Xp = (cl_float *) rbc.read (cl_algo::RBC::RBCConstruct<K, P>::Memory::H_OUT_X_P);

        // Extract permuted database
        idx = 0;
        std::generate (x.begin (), x.end (), [&]() { return Xp[4 * idx++]; });
        idx = 0;
        std::generate (y.begin (), y.end (), [&]() { return Xp[4 * idx++ + 1]; });

        // Compute representative radii
        for (int j = 0; j < NSIZE; ++j)
            if (ID[j].dist > radius[ID[j].id])
                radius[ID[j].id] = ID[j].dist;
        for (int j = 0; j < RNSIZE; ++j)
            radius[j] = std::sqrt (radius[j]);

        // Plot data points
        for (int j = 0; j < RNSIZE; ++j)
        {
            // pls->col0 (8);
            pls->col0 (randColor ());
            pls->ssym (0, 0.7);
            pls->poin (N[j], x.data () + O[j], y.data () + O[j], 17);
        }
        
        // Plot representative radii
        pls->col0 (12);
        for (int j = 0; j < RNSIZE; ++j)
        {
            // Circle
            pls->lsty (1);
            pls->arc (rx[j], ry[j], radius[j], radius[j], 0, 360, 90, 0);

            // Radius
            pls->lsty (2);
            PLFLT theta = pls->randd () * _2_PI;
            pls->join (rx[j], ry[j], rx[j] + radius[j] * std::cos (theta), ry[j] + radius[j] * std::sin (theta));
        }

        delete pls;  // Close PLplot library
    }

private:
    plstream *pls;

};


int main (int argc, const char **argv)
{
    try
    {
        plot rbc_2d (argc, argv);
    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }

    return 0;
}
