/*! \file rbc_kernels.cl
 *  \brief Kernels for building and accessing the Random Ball Cover data structure.
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


/*! \brief Computes \f$ \ell_1 \f$-norm based distances, 
 *         \f$ d(x,y)=\|x-y\|_1=\sum_{i}|x_i-y_i| \f$.
 *  \details Computes a 4x4 block of distances between the two input sets.
 *            
 *  \param[in] x array of database points holding only 4 of their dimensions.
 *  \param[in] r array of 4 representative points holding only 4 of their dimensions.
 *  \param[out] dists array of distances. Each row contains the distances of a 
 *                    database point from all 4 of the representative points.
 *  \param[in] n number of database points.
 */
inline
void l1NormMetric (float4 *x, float4 *r, float4 *dists, uint n)
{
    for (uint i = 0; i < n; ++i)
    {
        dists[i].x += dot (fabs (x[i] - r[0]), (float4) (1.f));
        dists[i].y += dot (fabs (x[i] - r[1]), (float4) (1.f));
        dists[i].z += dot (fabs (x[i] - r[2]), (float4) (1.f));
        dists[i].w += dot (fabs (x[i] - r[3]), (float4) (1.f));
    }
}


/*! \brief Computes \f$ \ell_2 \f$-norm based squared distances, 
 *         \f$ d(x,y)=\|x-y\|^2_2=\sum_{i}(x_i-y_i)^{2} \f$.
 *  \details Computes a 4x4 block of distances between the two input sets.
 *            
 *  \param[in] x array of database points holding only 4 of their dimensions.
 *  \param[in] r array of 4 representative points holding only 4 of their dimensions.
 *  \param[out] dists array of distances. Each row contains the distances of a 
 *                    database point from all 4 of the representative points.
 *  \param[in] n number of database points.
 */
inline
void euclideanSquaredMetric (float4 *x, float4 *r, float4 *dists, uint n)
{
    for (uint i = 0; i < n; ++i)
    {
        dists[i].x += dot (pown (x[i] - r[0], 2), (float4) (1.f));
        dists[i].y += dot (pown (x[i] - r[1], 2), (float4) (1.f));
        dists[i].z += dot (pown (x[i] - r[2], 2), (float4) (1.f));
        dists[i].w += dot (pown (x[i] - r[3], 2), (float4) (1.f));
    }
}


/*! \brief Computes the distances between two sets of points in a brute force way.
 *  \details For every point in the first set, the distances from that point to 
 *           all points in the second set are computed.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be equal 
 *        to the number of points in the second set, \f$ n \f$, divided by 4. That is, 
 *        \f$ gXdim = n/4 \f$. The **y** dimension of the global workspace, \f$ gYdim \f$, 
 *        should be equal to the number of points in the first set, \f$ m \f$, divided by 4. 
 *        That is, \f$ gYdim = m/4 \f$. The local workspace is irrelevant.
 *  \note Each **work-item** computes a **4x4 block** of output elements. The 
 *        **number of points** in each set should be a **multiple of 4**.
 *  \note The **dimensionality of the points** should be a **multiple of 4**. This restriction 
 *        can be avoided by handling the input data as `float`.
 *  \note The names of the variables in the kernel are specialized for a particular 
 *        use case (RBC construction). The functionality of the kernel remains generic.
 *  \attention The kernel doesn't use any shared memory for staging data from X and R. 
 *             There are other two cases implemented, `rbcComputeDists_SharedR` and
 *             `rbcComputeDists_SharedXR`. The access pattern for both arrays is strided,
 *             but the kernel's performance might actually be better than
 *             the other two cases', due to higher kernel occupancy.
 *            
 *  \param[in] X array of the database points (each row contains a point), \f$ X_{n_x \times d} \f$.
 *  \param[in] R array of the representative points (each row contains a point), \f$ R_{n_r \times d} \f$.
 *  \param[out] D array of distances of the database points from the representative
 *                points (each row contains the distances of a database point from 
 *                all the representative points), \f$ D_{n_x \times n_r} \f$.
 *  \param[in] d dimensionality of the associated points.
 */
kernel
void rbcComputeDists_SharedNone (global float4 *X, global float4 *R, global float4 *D, uint d)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    
    float4 dists[4];
    dists[0] = (float4) (0.f);
    dists[1] = (float4) (0.f);
    dists[2] = (float4) (0.f);
    dists[3] = (float4) (0.f);

    uint d4 = d >> 2;

    uint gPosX = (gY << 2) * d4;
    uint gPosR = (gX << 2) * d4;

    // Walk through the dimensions within the block, and compute 
    // intermediate results for the 4x4 block of the (gX, gY) work-item
    for (int j = 0; j < d4; ++j)
    {
        float4 x[4];
        x[0] = X[gPosX + j];
        x[1] = X[gPosX + d4 + j];
        x[2] = X[gPosX + 2 * d4 + j];
        x[3] = X[gPosX + 3 * d4 + j];

        float4 r[4];
        r[0] = R[gPosR + j];
        r[1] = R[gPosR + d4 + j];
        r[2] = R[gPosR + 2 * d4 + j];
        r[3] = R[gPosR + 3 * d4 + j];

        // Compute distances
        // l1NormMetric (x, r, dists, 4);
        euclideanSquaredMetric (x, r, dists, 4);
    }

    uint gPos = (gY << 2) * gXdim + gX;

    // Store the 4x4 block of computed 
    // distances in the output array
    D[gPos]             = dists[0];
    D[gPos + gXdim]     = dists[1];
    D[gPos + 2 * gXdim] = dists[2];
    D[gPos + 3 * gXdim] = dists[3];
}


/*! \brief Computes the distances between two sets of points in a brute force way.
 *  \details For every point in the first set, the distances from that point to 
 *           all points in the second set are computed.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be equal 
 *        to the number of points in the second set, \f$ n \f$, divided by 4. That is, 
 *        \f$ gXdim = n/4 \f$. The **y** dimension of the global workspace, \f$ gYdim \f$, 
 *        should be equal to the number of points in the first set, \f$ m \f$, divided by 4. 
 *        That is, \f$ gYdim = m/4 \f$. The local workspace should be square (performance 
 *        requirement). That is, \f$ lXdim = lYdim \f$.
 *  \note Each **work-item** computes a **4x4 block** of output elements. The **number of points** 
 *        in each set should be a **multiple of** \f$ 4*lXdim \f$. This restriction can be 
 *        relaxed to a multiple of \f$ lXdim \f$ by assigning one work-item per output element.
 *  \note The **dimensionality of the points** should be a **multiple of 4**. This restriction 
 *        can be avoided by handling the input data as `float`.
 *  \note The names of the variables in the kernel are specialized for a particular 
 *        use case (RBC construction). The functionality of the kernel remains generic.
 *  \attention The kernel uses shared memory for staging blocks of data just from R. 
 *             There are other two cases implemented, `rbcComputeDists_SharedNone` and
 *             `rbcComputeDists_SharedXR`. The kernel might have better performance than
 *             the last case, due to higher kernel occupancy.
 *            
 *  \param[in] X array of the database points (each row contains a point), \f$ X_{n_x \times d} \f$.
 *  \param[in] R array of the representative points (each row contains a point), \f$ R_{n_r \times d} \f$.
 *  \param[out] D array of distances of the database points from the representative
 *                points (each row contains the distances of a database point from 
 *                all the representative points), \f$ D_{n_x \times n_r} \f$.
 *  \param[in] data local buffer. Its size should be `16 float` elements for each 
 *                  work-item in a work-group. That is, \f$ (4*lXdim)*(4*lYdim)*sizeof\ (float) \f$.
 *  \param[in] d dimensionality of the associated points.
 */
kernel
void rbcComputeDists_SharedR (global float4 *X, global float4 *R, 
                              global float4 *D, local float4 *data, uint d)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint lY = get_local_id (1);
    
    float4 dists[4];
    dists[0] = (float4) (0.f);
    dists[1] = (float4) (0.f);
    dists[2] = (float4) (0.f);
    dists[3] = (float4) (0.f);

    uint d4 = d >> 2;

    //~ Dimensions are processed block by block
    // Compute the number of blocks necessary to process all dimensions
    uint dBlockIterations = (d4 / lXdim) + (d4 % lXdim > 0);
    // Compute the number of dimensions left in the last block
    uint dLast = select (lXdim, d4 % lXdim, d4 % lXdim);

    // Walk through the dimensions block by block
    for (int i = 0; i < dBlockIterations; ++i)
    {
        // R access pattern is strided, so
        // load blocks of data into local memory
        if ((i < (dBlockIterations - 1)) || (lX < dLast))
        {
            uint gPosR = (get_group_id (0) * (lXdim << 2) + (lY << 2)) * d4 + i * lXdim + lX;
            uint lPos  = (lY << 2) * lXdim + lX;

            data[lPos]             = R[gPosR];
            data[lPos + lXdim]     = R[gPosR + d4];
            data[lPos + 2 * lXdim] = R[gPosR + 2 * d4];
            data[lPos + 3 * lXdim] = R[gPosR + 3 * d4];
        }

        barrier (CLK_LOCAL_MEM_FENCE);

        uint gPosX = (gY << 2) * d4 + i * lXdim;
        uint dIterations = select (lXdim, dLast, i == (dBlockIterations - 1));

        // Walk through the dimensions within the block, and compute 
        // intermediate results for the 4x4 block of the (gX, gY) work-item
        for (int j = 0; j < dIterations; ++j)
        {
            uint lPosR = (lX << 2) * lXdim + j;

            float4 x[4];
            x[0] = X[gPosX + j];
            x[1] = X[gPosX + d4 + j];
            x[2] = X[gPosX + 2 * d4 + j];
            x[3] = X[gPosX + 3 * d4 + j];

            float4 r[4];
            r[0] = data[lPosR];
            r[1] = data[lPosR + lXdim];
            r[2] = data[lPosR + 2 * lXdim];
            r[3] = data[lPosR + 3 * lXdim];

            // Compute distances
            // l1NormMetric (x, r, dists, 4);
            euclideanSquaredMetric (x, r, dists, 4);
        }
        
        barrier (CLK_LOCAL_MEM_FENCE);        
    }

    uint gPos = (gY << 2) * gXdim + gX;

    // Store the 4x4 block of computed 
    // distances in the output array
    D[gPos]             = dists[0];
    D[gPos + gXdim]     = dists[1];
    D[gPos + 2 * gXdim] = dists[2];
    D[gPos + 3 * gXdim] = dists[3];
}


/*! \brief Computes the distances between two sets of points in a brute force way.
 *  \details For every point in the first set, the distances from that point to 
 *           all points in the second set are computed.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be equal 
 *        to the number of points in the second set, \f$ n \f$, divided by 4. That is, 
 *        \f$ gXdim = n/4 \f$. The **y** dimension of the global workspace, \f$ gYdim \f$, 
 *        should be equal to the number of points in the first set, \f$ m \f$, divided by 4. 
 *        That is, \f$ gYdim = m/4 \f$. The local workspace should be square (performance 
 *        requirement). That is, \f$ lXdim = lYdim \f$.
 *  \note Each **work-item** computes a **4x4 block** of output elements. The **number of points** 
 *        in each set should be a **multiple of** \f$ 4*lXdim \f$. This restriction can be 
 *        relaxed to a multiple of \f$ lXdim \f$ by assigning one work-item per output element.
 *  \note The **dimensionality of the points** should be a **multiple of 4**. This restriction 
 *        can be avoided by handling the input data as `float`.
 *  \note The names of the variables in the kernel are specialized for a particular 
 *        use case (RBC construction). The functionality of the kernel remains generic.
 *  \attention The kernel uses shared memory for staging blocks of data from both X and R. 
 *             There are other two cases implemented, `rbcComputeDists_SharedNone` and
 *             `rbcComputeDists_SharedR`. The kernel might have worse performance than
 *             those cases, due to high use of VGPRs and LDS, resulting in low kernel
 *             occupancy.
 *            
 *  \param[in] X array of the database points (each row contains a point), \f$ X_{n_x \times d} \f$.
 *  \param[in] R array of the representative points (each row contains a point), \f$ R_{n_r \times d} \f$.
 *  \param[out] D array of distances of the database points from the representative
 *                points (each row contains the distances of a database point from 
 *                all the representative points), \f$ D_{n_x \times n_r} \f$.
 *  \param[in] dataX local buffer. Its size should be `16 float` elements for each 
 *                   work-item in a work-group. That is, \f$ (4*lXdim)*(4*lYdim)*sizeof\ (float) \f$.
 *  \param[in] dataR local buffer. Its size should be `16 float` elements for each 
 *                   work-item in a work-group. That is, \f$ (4*lXdim)*(4*lYdim)*sizeof\ (float) \f$.
 *  \param[in] d dimensionality of the associated points.
 */
kernel
void rbcComputeDists_SharedXR (global float4 *X, global float4 *R, global float4 *D, 
                               local float4 *dataX, local float4 *dataR, uint d)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint lY = get_local_id (1);
    
    float4 dists[4];
    dists[0] = (float4) (0.f);
    dists[1] = (float4) (0.f);
    dists[2] = (float4) (0.f);
    dists[3] = (float4) (0.f);

    uint d4 = d >> 2;

    //~ Dimensions are processed block by block
    // Compute the number of blocks necessary to process all dimensions
    uint dBlockIterations = (d4 / lXdim) + (d4 % lXdim > 0);
    // Compute the number of dimensions left in the last block
    uint dLast = select (lXdim, d4 % lXdim, d4 % lXdim);

    // Walk through the dimensions block by block
    for (int i = 0; i < dBlockIterations; ++i)
    {
        // Both X and R access patterns are strided, so
        // load blocks of data from both X and R into local memory
        if ((i < (dBlockIterations - 1)) || (lX < dLast))
        {
            uint gPosX = (gY << 2) * d4 + i * lXdim + lX;
            uint gPosR = (get_group_id (0) * (lXdim << 2) + (lY << 2)) * d4 + i * lXdim + lX;
            uint lPos  = (lY << 2) * lXdim + lX;

            dataX[lPos]             = X[gPosX];
            dataX[lPos + lXdim]     = X[gPosX + d4];
            dataX[lPos + 2 * lXdim] = X[gPosX + 2 * d4];
            dataX[lPos + 3 * lXdim] = X[gPosX + 3 * d4];

            dataR[lPos]             = R[gPosR];
            dataR[lPos + lXdim]     = R[gPosR + d4];
            dataR[lPos + 2 * lXdim] = R[gPosR + 2 * d4];
            dataR[lPos + 3 * lXdim] = R[gPosR + 3 * d4];
        }

        barrier (CLK_LOCAL_MEM_FENCE);

        uint dIterations = select (lXdim, dLast, i == (dBlockIterations - 1));

        // Walk through the dimensions within the block, and compute 
        // intermediate results for the 4x4 block of the (gX, gY) work-item
        for (int j = 0; j < dIterations; ++j)
        {
            uint lPosX = (lY << 2) * lXdim + j;
            uint lPosR = (lX << 2) * lXdim + j;

            float4 x[4];
            x[0] = dataX[lPosX];
            x[1] = dataX[lPosX + lXdim];
            x[2] = dataX[lPosX + 2 * lXdim];
            x[3] = dataX[lPosX + 3 * lXdim];

            float4 r[4];
            r[0] = dataR[lPosR];
            r[1] = dataR[lPosR + lXdim];
            r[2] = dataR[lPosR + 2 * lXdim];
            r[3] = dataR[lPosR + 3 * lXdim];

            // Compute distances
            // l1NormMetric (x, r, dists, 4);
            euclideanSquaredMetric (x, r, dists, 4);
        }
        
        barrier (CLK_LOCAL_MEM_FENCE);        
    }

    uint gPos = (gY << 2) * gXdim + gX;

    // Store the 4x4 block of computed 
    // distances in the output array
    D[gPos]             = dists[0];
    D[gPos + gXdim]     = dists[1];
    D[gPos + 2 * gXdim] = dists[2];
    D[gPos + 3 * gXdim] = dists[3];
}


/*! \brief Computes the distances between two sets of points in a brute force way.
 *  \details For every point in the first set, the distances from that point to 
 *           all points in the second set are computed.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be equal 
 *        to the number of points in the second set, \f$ n \f$, divided by 4. That is, 
 *        \f$ gXdim = n/4 \f$. The **y** dimension of the global workspace, \f$ gYdim \f$, 
 *        should be equal to the number of points in the first set, \f$ m \f$, divided by 4. 
 *        That is, \f$ gYdim = m/4 \f$. The local workspace should be 
 *        \f$ (lXdim=8,lYdim=maxLocalSize/8) \f$ for optimal performance. 
 *        In general, it can be \f$ lXdim \leq lYdim \f$.
 *  \note Each **work-item** computes a **4x4 block** of output elements. The **number of points** 
 *        in the first set should be a **multiple of** \f$ 4*lYdim \f$. The **number of points** 
 *        in the second set should be a **multiple of** \f$ 4*lXdim \f$. This restriction can be 
 *        relaxed to a multiple of \f$ lYdim \f$ and \f$ lXdim \f$, respectively, by assigning 
 *        one work-item per output element.
 *  \note The **dimensionality of the points** should be a **multiple of 4**. This restriction 
 *        can be avoided by handling the input data as `float`.
 *  \note The names of the variables in the kernel are specialized for a particular 
 *        use case (RBC construction). The functionality of the kernel remains generic.
 *  \attention This is a specialization of the `rbcComputeDists_SharedXR` for the case of Kinect 
 *             8-D data `[geometric (Homogeneous coordinates) and photometric (RGBA values) information]`.
 *            
 *  \param[in] X array of the database points (each row contains a point), \f$ X_{n_x \times d} \f$.
 *  \param[in] R array of the representative points (each row contains a point), \f$ R_{n_r \times d} \f$.
 *  \param[out] D array of distances of the database points from the representative
 *                points (each row contains the distances of a database point from 
 *                all the representative points), \f$ D_{n_x \times n_r} \f$.
 *  \param[in] dataX local buffer. Its size should be `16 float` elements for each 
 *                   work-item in a work-group. That is, \f$ d*(4*lYdim)*sizeof\ (float) \f$.
 *  \param[in] dataR local buffer. Its size should be `16 float` elements for each 
 *                   work-item in a work-group. That is, \f$ d*(4*lYdim)*sizeof\ (float) \f$.
 */
kernel
void rbcComputeDists_Kinect (global float4 *X, global float4 *R, global float4 *D, 
                             local float4 *dataX, local float4 *dataR)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint lXdim = get_local_size (0);
    uint lYdim = get_local_size (1);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint lY = get_local_id (1);
    
    float4 dists[4];
    dists[0] = (float4) (0.f);
    dists[1] = (float4) (0.f);
    dists[2] = (float4) (0.f);
    dists[3] = (float4) (0.f);

    uint d4 = 2;  // d >> 2

    uint idx = lY * lXdim + lX;
    uint xi = idx % d4;
    uint yi = idx / d4;

    // Both X and R access patterns are strided, so
    // load blocks of data from both X and R into local memory
    if (idx < d4 * lYdim)
    {
        uint gPosX = (get_group_id (1) * (lYdim << 2) + (yi << 2)) * d4 + xi;
        uint gPosR = (get_group_id (0) * (lXdim << 2) + (yi << 2)) * d4 + xi;
        uint lPos  = (yi << 2) * d4 + xi;

        dataX[lPos]          = X[gPosX];
        dataX[lPos + d4]     = X[gPosX + d4];
        dataX[lPos + 2 * d4] = X[gPosX + 2 * d4];
        dataX[lPos + 3 * d4] = X[gPosX + 3 * d4];

        dataR[lPos]          = R[gPosR];
        dataR[lPos + d4]     = R[gPosR + d4];
        dataR[lPos + 2 * d4] = R[gPosR + 2 * d4];
        dataR[lPos + 3 * d4] = R[gPosR + 3 * d4];
    }

    barrier (CLK_LOCAL_MEM_FENCE);

    // Walk through the dimensions within the block, and compute 
    // intermediate results for the 4x4 block of the (gX, gY) work-item
    for (int j = 0; j < d4; ++j)
    {
        uint lPosX = (lY << 2) * d4 + j;
        uint lPosR = (lX << 2) * d4 + j;

        float4 x[4];
        x[0] = dataX[lPosX];
        x[1] = dataX[lPosX + d4];
        x[2] = dataX[lPosX + 2 * d4];
        x[3] = dataX[lPosX + 3 * d4];

        float4 r[4];
        r[0] = dataR[lPosR];
        r[1] = dataR[lPosR + d4];
        r[2] = dataR[lPosR + 2 * d4];
        r[3] = dataR[lPosR + 3 * d4];

        // Compute distances
        // l1NormMetric (x, r, dists, 4);
        euclideanSquaredMetric (x, r, dists, 4);
    }

    uint gPos = (gY << 2) * gXdim + gX;

    // Store the 4x4 block of computed 
    // distances in the output array
    D[gPos]             = dists[0];
    D[gPos + gXdim]     = dists[1];
    D[gPos + 2 * gXdim] = dists[2];
    D[gPos + 3 * gXdim] = dists[3];
}


// / *! \brief Computes the distances between two sets of points in a brute force way.
//  *  \note Same implementation as above. Only difference is that the transfer of the 
//  *        data into local memory is spread across double the work-items.
//  *        It is a bit slower due to all the extra computation.
//  */
// kernel
// void rbcComputeDists_Kinect_2 (global float4 *X, global float4 *R, global float4 *D, 
//                                local float4 *dataX, local float4 *dataR)
// {
//     // Workspace dimensions
//     uint gXdim = get_global_size (0);
//     uint lXdim = get_local_size (0);
//     uint lYdim = get_local_size (1);

//     // Workspace indices
//     uint gX = get_global_id (0);
//     uint gY = get_global_id (1);
//     uint lX = get_local_id (0);
//     uint lY = get_local_id (1);
    
//     float4 dists[4];
//     dists[0] = (float4) (0.f);
//     dists[1] = (float4) (0.f);
//     dists[2] = (float4) (0.f);
//     dists[3] = (float4) (0.f);

//     uint d4 = 2;  // d >> 2

//     uint idx = lY * lXdim + lX;
//     uint xi = idx % d4;
//     uint yi = idx / d4;

//     // Both X and R access patterns are strided, so
//     // load blocks of data from both X and R into local memory
//     if (idx < d4 * (lYdim << 1))
//     {
//         global float4 *gPtr[2] = { X, R };
//         local  float4 *lPtr[2] = { dataX, dataR };

//         uint XRflag = (idx / (d4 * lYdim)) == 1;

//         global float4  *In = gPtr[XRflag];
//         local  float4 *Out = lPtr[XRflag];

//         uint gPos = select (
//             (get_group_id (1) * (lYdim << 2) + (yi << 2)) * d4 + xi,
//             (get_group_id (0) * (lXdim << 2) + ((yi - lYdim) << 2)) * d4 + xi,
//             XRflag);

//         uint lPos = select (
//             (yi << 2) * d4 + xi, 
//             ((yi - lYdim) << 2) * d4 + xi, 
//             XRflag);

//         Out[lPos]          = In[gPos];
//         Out[lPos + d4]     = In[gPos + d4];
//         Out[lPos + 2 * d4] = In[gPos + 2 * d4];
//         Out[lPos + 3 * d4] = In[gPos + 3 * d4];
//     }

//     barrier (CLK_LOCAL_MEM_FENCE);

//     // Walk through the dimensions within the block, and compute 
//     // intermediate results for the 4x4 block of the (gX, gY) work-item
//     for (int j = 0; j < d4; ++j)
//     {
//         uint lPosX = (lY << 2) * d4 + j;
//         uint lPosR = (lX << 2) * d4 + j;

//         float4 x[4];
//         x[0] = dataX[lPosX];
//         x[1] = dataX[lPosX + d4];
//         x[2] = dataX[lPosX + 2 * d4];
//         x[3] = dataX[lPosX + 3 * d4];

//         float4 r[4];
//         r[0] = dataR[lPosR];
//         r[1] = dataR[lPosR + d4];
//         r[2] = dataR[lPosR + 2 * d4];
//         r[3] = dataR[lPosR + 3 * d4];

//         // Compute distances
//         // l1NormMetric (x, r, dists, 4);
//         euclideanSquaredMetric (x, r, dists, 4);
//     }

//     uint gPos = (gY << 2) * gXdim + gX;

//     // Store the 4x4 block of computed 
//     // distances in the output array
//     D[gPos]             = dists[0];
//     D[gPos + gXdim]     = dists[1];
//     D[gPos + 2 * gXdim] = dists[2];
//     D[gPos + 3 * gXdim] = dists[3];
// }


/*! \brief Performs an array initialization.
 *  \details Initializes an 1-D array with a provided value.
 *  \note The number of elements, `n`, in the array should be a **multiple of 4** 
 *        (the data are handled as `uint4`). The global workspace should be one 
 *        dimensional. The **x** dimension of the global workspace, \f$ gXdim \f$, 
 *        should be equal to the number of elements in the array divided by 4. 
 *        That is, \f$ \ gXdim=n/4 \f$. The local workspace is irrelevant.
 *
 *  \param[out] N array that is going to contain the cardinalities of the 
 *                representative lists. Its size should be \f$ n*sizeof\ (uint) \f$.
 *  \param[in] val initialization value.
 */
kernel
void rbcNInit (global uint4 *N, uint val)
{
    uint gX = get_global_id (0);

    N[gX] = (uint4) (val);
}


/*! \brief Struct holding a value and a key.
 *  \details Data structure used during the reduction phase of the distances 
 *           in the array produced by a `rbcComputeDists` kernel.
 */
typedef struct
{
    float dist;
    uint id;
} dist_id;


/*! \brief Performs a reduce operation on the columns of an array.
 *  \details Computes the minimum element, and its corresponding column id, for each 
 *           row in an array. It also builds a histogram of the id values. And lastly, 
 *           it stores the rank (order of insert) of each minimum element within its 
 *           corresponding histogram bin.
 *  \note When there are multiple rows in the array, a reduce operation 
 *        is performed per row, in parallel.
 *  \note The number of elements, `n`, in a row of the array should be a **multiple 
 *        of 4** (the data are handled as `float4`). The **x** dimension of the 
 *        global workspace, \f$ gXdim \f$, should be greater or equal to the number of 
 *        elements in a row of the array divided by 8. That is, \f$ \ gXdim \geq n/8 \f$. 
 *        Each work-item handles `8 float` (= `2 float4`) elements in a row of the array. 
 *        The **y** dimension of the global workspace, \f$ gYdim \f$, should be equal 
 *        to the number of rows, `m`, in the array. That is, \f$ \ gYdim = m \f$. 
 *        The local workspace should be `1` in the **y** dimension, and a 
 *        **power of 2** in the **x** dimension. It is recommended 
 *        to use one `wavefront/warp` per work-group.
 *  \note When the number of elements per row of the array is small enough to be 
 *        handled by a single work-group, the output array, `ID`, will contain the 
 *        true minimums. When the elements are more than that, they are partitioned 
 *        into blocks and reduced independently. In this case, the kernel outputs 
 *        the minimums from each block reduction. A reduction should then be
 *        made on those minimums for the final results by dispatching the 
 *        `rbcGroupMinDists` kernel.
 *  \note The kernel increments the `N` counters. `rbcNListInit` should be called, 
 *        before `rbcMinDists` is dispatched, in order to initialize the counters.
 *
 *  \param[in] D input array of `float` elements.
 *  \param[out] ID (reduced) output array of `dist_id` elements. When the kernel is
 *                 dispatched with one work-group per row, the array contains the 
 *                 final results, and its size should be \f$ m*sizeof\ (dist\_id) \f$.
 *                 When the kernel is dispatched with more than one work-group per row,
 *                 the array contains the results from each block reduction, and its 
 *                 size should be \f$ wgXdim*m*sizeof\ (dist\_id) \f$.
 *  \param[in] data local buffer. Its size should be `2 dist_id` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (dist\_id) \f$.
 *  \param[out] N array containing the cardinalities of the representative lists.
 *                Its size should be \f$ n*sizeof\ (uint) \f$.
 *  \param[out] Rnk array containing the rank (aka order, index) of each database 
 *                  point within the associated representative list. Its size should 
 *                  be \f$ m*sizeof\ (uint) \f$.
 *  \param[in] n number of elements in a row of the array divided by 4.
 *  \param[in] accCounters a flag to indicate whether or not to involve in the computation 
 *                         the list element counters, `N`, and element ranks, `Rnk`.
 */
kernel
void rbcMinDists (global float4 *D, global dist_id *ID, 
                  global uint *N, global uint *Rnk, 
                  local dist_id *data, uint n, int accCounters)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Load 8 float elements per work-item
    int4 flag = (uint4) (gX) < (uint4) (n);
    float4 a = select ((float4) (INFINITY), D[gY * n + gX], flag);
    flag = (uint4) (gX + lXdim) < (uint4) (n);
    float4 b = select ((float4) (INFINITY), D[gY * n + gX + lXdim], flag);

    // Base index (for the first float4
    // element handled by the work-item)
    uint idx_a = gX << 2;
    // Base index (for the second float4
    // element handled by the work-item)
    uint idx_b = (gX + lXdim) << 2;

    // Reduce a
    int   fa_lo = isless (a.x, a.y);
    float da_lo = select (a.y, a.x, fa_lo);
    uint  ia_lo = select (idx_a + 1, idx_a, fa_lo);
    int   fa_hi = isless (a.z, a.w);
    float da_hi = select (a.w, a.z, fa_hi);
    uint  ia_hi = select (idx_a + 3, idx_a + 2, fa_hi);
    int   fa_md = isless (da_lo, da_hi);
    float da_md = select (da_hi, da_lo, fa_md);
    uint  ia_md = select (ia_hi, ia_lo, fa_md);

    // Reduce b
    int   fb_lo = isless (b.x, b.y);
    float db_lo = select (b.y, b.x, fb_lo);
    uint  ib_lo = select (idx_b + 1, idx_b, fb_lo);
    int   fb_hi = isless (b.z, b.w);
    float db_hi = select (b.w, b.z, fb_hi);
    uint  ib_hi = select (idx_b + 3, idx_b + 2, fb_hi);
    int   fb_md = isless (db_lo, db_hi);
    float db_md = select (db_hi, db_lo, fb_md);
    uint  ib_md = select (ib_hi, ib_lo, fb_md);
    
    // Store the min of a and b
    data[lX].dist = da_md;
    data[lX].id = ia_md;
    data[lX + lXdim].dist = db_md;
    data[lX + lXdim].id = ib_md;

    // Reduce
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);

        if (lX < d)
        {
            dist_id da = data[lX];
            dist_id db = data[lX + d];

            int flag = isless (da.dist, db.dist);
            data[lX].dist = select (db.dist, da.dist, flag);
            data[lX].id = select (db.id, da.id, flag);
        }
    }

    // One work-item per work-group 
    // stores the minimum element
    if (lX == 0) 
        ID[gY * wgXdim + wgX] = data[0];

    // Handle counter and rank
    if (accCounters == 1 && gX == 0)
        Rnk[gY] = atomic_inc (&N[data[0].id]);
}


/*! \brief Performs a reduce operation on the columns of an array.
 *  \details Computes the minimum element, and its corresponding column id, for each 
 *           row in an array. It also builds a histogram of the id values. And lastly, 
 *           it stores the rank (order of insert) of each minimum element within its 
 *           corresponding histogram bin.
 *  \note When there are multiple rows in the array, a reduce operation 
 *        is performed per row, in parallel.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        greater or equal to the number of elements in a row of the array divided 
 *        by 2. That is, \f$ \ gXdim \geq n/2 \f$. Each work-item handles `2 dist_id` 
 *        elements in a row of the array. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `m`, in the array. 
 *        That is, \f$ \ gYdim = m \f$. The local workspace should be `1` in the **y** 
 *        dimension, and a **power of 2** in the **x** dimension. It is recommended 
 *        to use one `wavefront/warp` per work-group.
 *  \note When the number of elements per row of the array is small enough to be 
 *        handled by a single work-group, the output array will contain the true 
 *        minimums. When the elements are more than that, they are partitioned 
 *        into blocks and reduced independently. In this case, the kernel outputs 
 *        the minimums from each block reduction. A reduction should then be
 *        made on those minimums for the final results. 
 *  \note The kernel increments the `N` counters. `rbcNListInit` should be called, 
 *        before `rbcGroupMinDists` is dispatched, in order to initialize the counters.
 *
 *  \param[in] GM input array of `dist_id` elements.
 *  \param[out] ID (reduced) output array of `dist_id` elements. When the kernel is
 *                 dispatched with one work-group per row, the array contains the 
 *                 final results, and its size should be \f$ m*sizeof\ (dist\_id) \f$.
 *                 When the kernel is dispatched with more than one work-group per row,
 *                 the array contains the results from each block reduction, and its 
 *                 size should be \f$ wgXdim*m*sizeof\ (dist\_id) \f$.
 *  \param[in] data local buffer. Its size should be `2 dist_id` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (dist\_id) \f$.
 *  \param[out] N array containing the cardinalities of the representative lists.
 *                Its size should be \f$ n*sizeof\ (uint) \f$.
 *  \param[out] Rnk array containing the rank (aka order, index) of each database 
 *                  point within the associated representative list. Its size should 
 *                  be \f$ m*sizeof\ (uint) \f$.
 *  \param[in] n number of elements in a row of the array.
 *  \param[in] accCounters a flag to indicate whether or not to involve in the computation 
 *                         the list element counters, `N`, and element ranks, `Rnk`.
 */
kernel
void rbcGroupMinDists (global dist_id *GM, global dist_id *ID, 
                       global uint *N, global uint *Rnk, 
                       local dist_id *data, uint n, int accCounters)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    dist_id dummy = { INFINITY, (uint) -1 };

    // Load 2 dist_id elements per work-item
    data[lX] = (gX < n) ? GM[gY * n + gX] : dummy;
    data[lX + lXdim] = (gX + lXdim < n) ? GM[gY * n + gX + lXdim] : dummy;

    // Reduce
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);

        if (lX < d)
        {
            dist_id da = data[lX];
            dist_id db = data[lX + d];

            int flag = isless (da.dist, db.dist);
            data[lX].dist = select (db.dist, da.dist, flag);
            data[lX].id = select (db.id, da.id, flag);
        }
    }

    // One work-item per work-group 
    // stores the minimum element
    if (lX == 0)
        ID[gY * wgXdim + wgX] = data[0];

    // Handle counter and rank
    if (accCounters == 1 && gX == 0)
        Rnk[gY] = atomic_inc (&N[data[0].id]);
}


/*! \brief Performs a permutation of the `RBC` database.
 *  \details Permutes the database points to form the representative lists and 
 *           allow for coalesced access pattern during the search operation.
 *  \note The **dimensionality of the points**, `d`, should be a **multiple of 4**
 *        (the data are handled as `float4`). The **x** dimension of the global 
 *        workspace, \f$ gXdim \f$, should be equal to the dimensionality of the 
 *        points divided by 4. That is, \f$ \ gXdim=d/4 \f$. The **y** dimension 
 *        of the global workspace, \f$ gYdim \f$, should be equal to the number 
 *        of database points, \f$ n_x \f$. That is, \f$ \ gYdim = n_x \f$. 
 *        The local workspace is irrelevant.
 *
 *  \param[in] X array of the database points (each row contains a point), 
 *               \f$ X_{n_x \times d} \f$.
 *  \param[out] Xp permuted array of the database points (each row contains 
 *                 a point), \f$ X_{n_x \times d} \f$.
 *  \param[in] ID array with the minimum distances and representative ids per
 *                database point. Its size should be \f$ n_x*sizeof\ (dist\_id) \f$.
 *  \param[in] O array containing the offsets of the representative lists within 
 *               the permuted database. Its size should be \f$ n_r*sizeof\ (uint) \f$.
 *  \param[in] Rnk array containing the rank (aka order, index) of each database 
 *                 point within the associated representative list. Its size should 
 *                 be \f$ n_x*sizeof\ (uint) \f$.
 */
kernel
void rbcPermute (global float4 *X, global float4 *Xp, 
                 global dist_id *ID, global uint *O, global uint *Rnk)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    // Read (part of) the database point
    float4 p = X[gY * gXdim + gX];

    // Read the point's representative id
    uint id = ID[gY].id;

    // Read the representative list's offset within the database
    uint offset = O[id];

    // Read the point's index within the representative list
    uint rank = Rnk[gY];

    // Store the point in its representative list
    Xp[(offset + rank) * gXdim + gX] = p;
}


/*! \brief Implements the second step of the `RBC search` algorithm.
 *  \details Computes the distances of each query from its representative list, 
 *           and outputs the query's nearest neighbors.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be greater 
 *        than or equal to the number of points in the represetative list with the greatest 
 *        cardinality, divided by 4. That is, \f$ gXdim \geq max_{id_r}{N[id_r]}/4 \f$. The 
 *        **y** dimension of the global workspace, \f$ gYdim \f$, should be equal to the 
 *        number of queries, \f$ n_q \f$. That is, \f$ gYdim = n_q \f$. The local workspace 
 *        should be `1` in the **y** dimension, and a **power of 2** in the **x** dimension. 
 *        It is recommended to use one `wavefront/warp` per work-group.
 *  \note The **dimensionality of the points** should be a **multiple of 4**. This restriction 
 *        can be avoided by handling the input data as `float`.
 *  \note When the number of points per query, \f$ max_{id_r}{N[id_r]}/4 \f$, is small 
 *        enough to be handled by a single work-group, the output array, `RID`, will 
 *        contain the final NNs. When the points are more than that, they are partitioned 
 *        into blocks and reduced independently. In this case, the kernel outputs the 
 *        minimums from each block reduction. A reduction should then be made on those 
 *        minimums for the final NNs by dispatching the `rbcGroupMinDists` kernel.
 *  \attention In order for this kernel to be efficient there shouldn't be great load 
 *             imbalance. That is, it is assumed that the database points are uniformly 
 *             distributed across all representative lists. The alternative would be 
 *             to process each query one at a time. But dispatching the kernel 
 *             \f$ n_q \f$ times is not really a viable option.
 *            
 *  \param[in] Q array of the query points (each row contains a point), \f$ Q_{n_q \times d} \f$.
 *  \param[in] Xp array of the database points (each row contains a point), \f$ Xp_{n_x \times d} \f$.
 *  \param[in] O array containing the offsets of the representative lists within 
 *               the permuted database. Its size should be \f$ n_r*sizeof\ (uint) \f$.
 *  \param[in] N array containing the cardinalities of the representative lists.
 *               Its size should be \f$ n_r*sizeof\ (uint) \f$.
 *  \param[in] RID array with the representative ids for each query. Its size 
 *                 should be \f$ n_q*sizeof\ (dist\_id) \f$.
 *  \param[out] NNID output array of `dist_id` elements. When the kernel is dispatched 
 *                   with one work-group per row, the array contains the final results, 
 *                   and its size should be \f$ n_q*sizeof\ (dist\_id) \f$. When the 
 *                   kernel is dispatched with more than one work-group per row, the 
 *                   array contains the results from each block reduction, and its size 
 *                   should be \f$ wgXdim*n_q*sizeof\ (dist\_id) \f$.
 *  \param[in] data local buffer. Its size should be `2 dist_id` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (dist\_id) \f$.
 *  \param[in] d dimensionality of the associated points.
 */
kernel
void rbcComputeQXMinDists (global float4 *Q, global float4 *Xp, global uint *O, global uint *N, 
                           global dist_id *RID, global dist_id *NNID, local dist_id *data, uint d)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    uint d4 = d >> 2;       // Number of 4-dimensions
    uint rID = RID[gY].id;  // Representative ID
    uint o = O[rID];        // Offset of rID's list within the database
    uint n = N[rID];        // Number of points in rID's list

    global float4 *L = Xp + o * d4;  // rID's list

    // Compute distances =======================================================

    float4 dists = (float4) (0.f);

    uint gX4 = gX << 2;
    uint gPosQ = gY * d4;
    uint gPosL = gX4 * d4;

    int4 f0 = (uint4) (gX4)     < (uint4) (n);
    int4 f1 = (uint4) (gX4 + 1) < (uint4) (n);
    int4 f2 = (uint4) (gX4 + 2) < (uint4) (n);
    int4 f3 = (uint4) (gX4 + 3) < (uint4) (n);

    // Walk through the dimensions and compute intermediate 
    // results for the 1x4 block of the (gX, gY) work-item
    for (int j = 0; j < d4; ++j)
    {
        float4 q = Q[gPosQ + j];

        float4 l[4];
        l[0] = select ((float4) (INFINITY), L[gPosL + j], f0);
        l[1] = select ((float4) (INFINITY), L[gPosL + d4 + j], f1);
        l[2] = select ((float4) (INFINITY), L[gPosL + 2 * d4 + j], f2);
        l[3] = select ((float4) (INFINITY), L[gPosL + 3 * d4 + j], f3);

        // Compute distances
        // l1NormMetric (q, l, dists, 1);
        euclideanSquaredMetric (&q, l, &dists, 1);
    }

    // Reduce ==================================================================
    // The previous step uses 1 work-item per 4 database points
    // The current step needs 1 work-item per 2 database points
    // The data are going to be reduced in half per work-item, 
    // and then the usual reduction algorithm will be executed

    uint idx = o + (gX << 2);
    dist_id a, b;

    // Reduce float4 element
    int fa = isless (dists.x, dists.y);
    a.dist = select (dists.y, dists.x, fa);
    a.id   = select (idx + 1, idx    , fa);
    int fb = isless (dists.z, dists.w);
    b.dist = select (dists.w, dists.z, fb);
    b.id   = select (idx + 3, idx + 2, fb);

    data[2 * lX] = a;
    data[2 * lX + 1] = b;

    // Reduce
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);

        if (lX < d)
        {
            a = data[lX];
            b = data[lX + d];

            int flag = isless (a.dist, b.dist);
            data[lX].dist = select (b.dist, a.dist, flag);
            data[lX].id = select (b.id, a.id, flag);
        }
    }

    // One work-item per work-group 
    // stores the minimum element
    if (lX == 0)
        NNID[gY * wgXdim + wgX] = data[0];
}


/*! \brief Collects the query NNs into an array.
 *  \note The **dimensionality of the points**, `d`, should be a **multiple of 4**
 *        (the data are handled as `float4`). The **x** dimension of the global 
 *        workspace, \f$ gXdim \f$, should be equal to the dimensionality of the 
 *        points divided by 4. That is, \f$ \ gXdim=d/4 \f$. The **y** dimension 
 *        of the global workspace, \f$ gYdim \f$, should be equal to the number 
 *        of queries, \f$ n_q \f$. That is, \f$ \ gYdim = n_q \f$. 
 *        The local workspace is irrelevant.
 *
 *  \param[in] Xp permuted array of the database points (each row contains 
 *                a point), \f$ X_{n_x \times d} \f$.
 *  \param[out] NN array of queries' nearest neigbors (each row contains 
 *                 a point), \f$ X_{n_q \times d} \f$.
 *  \param[in] NNID array with the minimum distances and NN ids per query. 
 *                  Its size should be \f$ n_q*sizeof\ (dist\_id) \f$.
 */
kernel
void rbcGetNNs (global float4 *Xp, global float4 *NN, global dist_id *NNID)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    // Read the NN id
    uint id = NNID[gY].id;

    // Read (part of) the NN
    float4 p = Xp[id * gXdim + gX];

    // Store the NN
    NN[gY * gXdim + gX] = p;
}
