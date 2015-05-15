/*! \file helper_funcs.hpp
 *  \brief Declarations of helper functions for testing.
 *  \author Nick Lamprianidis
 *  \version 1.0
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

#ifndef HELPERFUNCS_HPP
#define HELPERFUNCS_HPP

#include <cassert>
#include <algorithm>
#include <functional>
#include <RBC/data_types.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


namespace RBC
{

    /*! \brief Checks the command line arguments for the profiling flag, `--profiling`. */
    bool setProfilingFlag (int argc, char **argv);


    /*! \brief Returns the first power of 2 greater than or equal to the input.
     *
     *  \param[in] num input data.
     *  \return The first power of 2 >= num.
     */
    template <typename T>
    uint64_t nextPow2 (T num)
    {
        assert (num >= 0);

        uint64_t pow;
        for (pow = 1; pow < (uint64_t) num; pow <<= 1) ;

        return pow;
    }


    /*! \brief Prints an array of an integer type to standard output.
     *
     *  \tparam T type of the data to be printed.
     *  \param[in] title legend for the output.
     *  \param[in] ptr array that is to be displayed.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void printBuffer (const char *title, T *ptr, uint32_t width, uint32_t height)
    {
        std::cout << title << std::endl;

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                std::cout << std::setw (3 * sizeof (T)) << +ptr[row * width + col] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    }


    /*! \brief Prints an array of floating-point type to standard output.
     *
     *  \tparam T type of the data to be printed.
     *  \param[in] title legend for the output.
     *  \param[in] ptr array that is to be displayed.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     *  \param[in] prec the number of decimal places to print.
     */
    template <typename T>
    void printBufferF (const char *title, T *ptr, uint32_t width, uint32_t height, uint32_t prec)
    {
        std::ios::fmtflags f (std::cout.flags ());
        std::cout << title << std::endl;
        std::cout << std::fixed << std::setprecision (prec);

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                std::cout << std::setw (5 + prec) << ptr[row * width + col] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
        std::cout.flags (f);
    }


    /*! \brief Reduces each row of an array to a single element.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input data.
     *  \param[out] out output (reduced) data.
     *  \param[in] cols number of columns in the input array.
     *  \param[in] rows number of rows in the input array.
     */
    template <typename T>
    void cpuReduce (T *in, T *out, uint32_t cols, uint32_t rows, std::function<bool (T, T)> func)
    {
        for (uint r = 0; r < rows; ++r)
        {
            T rec = in[r * cols];
            for (uint c = 1; c < cols; ++c)
            {
                T tmp = in[r * cols + c];
                if (func (tmp, rec)) rec = tmp;
            }
            out[r] = rec;
        }
    }


    /*! \brief Performs an inclusive scan operation on the columns of an array.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (scan) data.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void cpuInScan (T *in, T *out, uint32_t width, uint32_t height)
    {
        // Initialize the first element of each row
        for (uint32_t row = 0; row < height; ++row)
            out[row * width] = in[row * width];
        // Perform the scan
        for (uint32_t row = 0; row < height; ++row)
            for (uint32_t col = 1; col < width; ++col)
                out[row * width + col] = out[row * width + col - 1] + in[row * width + col];
    }


    /*! \brief Performs an exclusive scan operation on the columns of an array.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (scan) data.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void cpuExScan (T *in, T *out, uint32_t width, uint32_t height)
    {
        // Initialize the first element of each row
        for (uint32_t row = 0; row < height; ++row)
            out[row * width] = 0;
        // Perform the scan
        for (uint32_t row = 0; row < height; ++row)
            for (uint32_t col = 1; col < width; ++col)
                out[row * width + col] = out[row * width + col - 1] + in[row * width + col - 1];
    }


    /*! \brief Computes the distances between two sets of points in a brute force way.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] X array of the database points (each row contains a point).
     *  \param[in] R array of the representative points (each row contains a point).
     *  \param[out] D array of distances of the database points from the representative
     *                points (each row contains the distances of a database point from 
     *                all the representative points)
     *  \param[in] nx number of database points.
     *  \param[in] nr number of representative points.
     *  \param[in] d dimensionality of the associated points.
     */
    template <typename T>
    void cpuRBCComputeDists (T *X, T *R, T *D, uint32_t nx, uint32_t nr, uint32_t d)
    {
        for (uint x = 0; x < nx; ++x)
        {
            for (uint r = 0; r < nr; ++r)
            {
                float dist = 0.f;
                for (uint j = 0; j < d; ++j)
                    dist += std::pow (X[x * d + j] - R[r * d + j], 2);

                D[x * nr + r] = dist;
            }
        }
    }


    /*! \brief Computes the minimum element, and its corresponding column id, for each 
     *         row in an array. It also builds a histogram of the id values. And lastly, 
     *         it stores the rank (order of insert) of each minimum element within its 
     *         corresponding histogram bin.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input data.
     *  \param[out] out output (reduced) data (min and column id).
     *  \param[out] N array with the representative list cardinalities.
     *  \param[out] Rnk array with the indices of each database point within the associated list.
     *  \param[in] cols number of columns in the input array.
     *  \param[in] rows number of rows in the input array.
     *  \param[in] accCounters a flag to indicate whether or not to involve in the computation 
     *                         the list element counters, `N`, and element ranks, `Rnk`.
     */
    template <typename T>
    void cpuRBCMinDists (T *in, rbc_dist_id *out, uint32_t *N, uint32_t *Rnk, 
                         uint32_t cols, uint32_t rows, bool accCounters)
    {
        if (accCounters)
            for (uint c = 0; c < cols; ++c)
                N[c] = 0;

        rbc_dist_id rbcMin;

        for (uint r = 0; r < rows; ++r)
        {
            rbcMin.dist = in[r * cols];
            rbcMin.id = 0;

            for (uint c = 1; c < cols; ++c)
            {
                T tmp = in[r * cols + c];
                if (tmp < rbcMin.dist)
                {
                    rbcMin.dist = tmp;
                    rbcMin.id = c;
                }
            }

            out[r] = rbcMin;

            if (accCounters)
                Rnk[r] = N[rbcMin.id]++;
        }
    }


    /*! \brief Performs a permutation of the `RBC` database 
     *         to form the representative lists.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] X array of database points (each row contains a point).
     *  \param[in] ID array with the minimum distances and representative ids per database point.
     *  \param[out] Xp permuted database.
     *  \param[out] IDp array with the minimum distances and representative ids per database point in Xp.
     *  \param[in] O array containing the index (offset) of the first element of 
     *               each representative list within the database.
     *  \param[in] Rnk array containing the rank (aka order, index) of each database 
     *                 point within the associated representative list.
     *  \param[in] nx number of database points.
     *  \param[in] nr number of representative points.
     *  \param[in] d dimensionality of the associated points.
     *  \param[in] permID flag to indicate whether or not to also permute the ID array.
     */
    template <typename T>
    void cpuRBCPermute (T *X, rbc_dist_id *ID, T *Xp, rbc_dist_id *IDp, uint32_t *O, uint32_t *Rnk, 
                        uint32_t nx, uint32_t nr, uint32_t d, bool permID)
    {
        for (uint32_t x = 0; x < nx; ++x)
        {
            uint id = ID[x].id;
            uint offset = O[id];
            uint rank = Rnk[x];

            for (uint32_t j = 0; j < d; ++j)
                Xp[(offset + rank) * d + j] = X[x * d + j];

            if (permID == true)
                IDp[offset + rank] = ID[x];
        }
    }


    /*! \brief Uses the RBC data structure to search for the nearest neighbors.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] Qp permuted array of query points (each row contains a point).
     *  \param[in] RID array with minimum distances and representative ids per query point in Qp.
     *  \param[in] Xp permuted array of database points (each row contains a point).
     *  \param[in] O array containing the index (offset) of the first element of 
     *               each representative list within the database.
     *  \param[in] N array with the representative list cardinalities.
     *  \param[out] NNID array with distances and NN ids for each query point.
     *  \param[out] NN array of NN points (each row contains a point).
     *  \param[in] nq number of query points.
     *  \param[in] nr number of representative points.
     *  \param[in] nx number of database points.
     *  \param[in] d dimensionality of the associated points.
     */
    template <typename T>
    void cpuRBCSearch (T *Qp, rbc_dist_id *RID, T *Xp, cl_uint *O, cl_uint *N, rbc_dist_id *NNID, T *NN, 
                       uint32_t nq, uint32_t nr, uint32_t nx, uint32_t d)
    {
        cl_uint max_n = std::accumulate (N, N + nr, 0, 
            [](cl_uint a, cl_uint b) -> cl_uint { return std::max (a, b); });

        cl_float *D = new cl_float[nq * max_n];
        
        // Compute distances Q - X[L]
        for (uint q = 0; q < nq; ++q)
        {
            cl_uint rID = RID[q].id;
            cl_uint o = O[rID];
            cl_uint n = N[rID];

            for (uint x = 0; x < max_n; ++x)
            {
                float dist = 0.f;

                if (x < n)
                    for (uint j = 0; j < d; ++j)
                        dist += std::pow (Qp[q * d + j] - Xp[(o + x) * d + j], 2);
                else
                    dist = std::numeric_limits<float>::infinity ();

                D[q * max_n + x] = dist;
            }
        }
        
        cpuRBCMinDists (D, NNID, nullptr, nullptr, max_n, nq, false);

        for (uint32_t q = 0; q < nq; ++q)
        {
            uint nnID = O[RID[q].id] + NNID[q].id;

            for (uint32_t j = 0; j < d; ++j)
                NN[q * d + j] = Xp[nnID * d + j];
        }

        delete[] D;
    }


    /*! \brief Calculates the euclidean distance betweeen two points.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] p1 first point.
     *  \param[in] p2 second point.
     *  \param[in] d dimensionality of the associated points.
     */
    template <typename T>
    T euclideanMetric (T *p1, T *p2, uint32_t d)
    {
        T dist = 0.f;

        for (int i = 0; i < d; ++i)
            dist += std::pow (p1[i] - p2[i], 2);

        return std::sqrt (dist);
    }


    /*! \brief Computes (brute force) the nearest neighbors of a set of queries.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] Q array of query points (each row contains a point).
     *  \param[in] X array of database points (each row contains a point).
     *  \param[out] NN array of NN points (each row contains a point).
     *  \param[in] nq number of query points.
     *  \param[in] nx number of database points.
     *  \param[in] d dimensionality of the associated points.
     */
    template <typename T>
    void cpuNNSearch (T *Q, T *X, T *NN, uint32_t nq, uint32_t nx, uint32_t d)
    {
        // Compute distance array
        std::vector<T> D (nq * nx);

        for (uint32_t q = 0; q < nq; ++q)
            for (uint32_t x = 0; x < nx; ++x)
                D[q * nx + x] = euclideanMetric (&Q[q * d], &X[x * d], d);

        // Find NN IDs
        std::vector<uint32_t> minID (nq);

        for (uint32_t q = 0; q < nq; ++q)
        {
            T minDist = D[q * nx];
            uint32_t id = 0;

            for (uint32_t x = 1; x < nx; ++x)
            {
                T tmp = D[q * nx + x];
                if (tmp < minDist)
                {
                    minDist = tmp;
                    id = x;
                }
            }

            minID[q] = id;
        }

        // Collect NNs
        for (uint32_t q = 0; q < nq; ++q)
            for (uint32_t j = 0; j < d; ++j)
                NN[q * d + j] = X[minID[q] * d + j];     
    }


    /*! \brief Computes the mean euclidean distance from the queries to their NNs.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] Q array of queries (each row contains a point).
     *  \param[in] NN array of NNs (each row contains a point).
     *  \param[in] n number of points in the arrays.
     *  \param[in] d dimensionality of the associated points.
     */
    template <typename T>
    T meanError (T *Q, T *NN, uint32_t n, uint32_t d)
    {
        std::vector<T> D (n);
        unsigned int i = -1;

        std::generate (D.begin (), D.end (), 
            [&]() { i++; return euclideanMetric (&Q[i * d], &NN[i * d], d); });

        T sumD = std::accumulate (D.begin (), D.end (), 0.f);

        return sumD / (T) n;
    }

}

#endif  // HELPERFUNCS_HPP
