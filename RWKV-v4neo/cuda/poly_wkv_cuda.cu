#include <stdio.h>
#include <assert.h>

#define MIN_VALUE (-1e38)
#define CONTEXT_LENGTH 1e5
#define CURVES 20

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                               F *__restrict__ const _y)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    F decay[CURVES], curve_coeff[CURVES];
    F aa[CURVES], bb[CURVES];
    F pp = MIN_VALUE;
    for (int j = 0; j < CURVES; j++)
    {
        F fraction = (F)j / (F)(CURVES - 1);
        F evaluation_point = exp((2 * fraction - 1) * log(CONTEXT_LENGTH));
        decay[j] = exp(-1.0 / evaluation_point);
        curve_coeff[j] = -1.0 / evaluation_point + log(evaluation_point) * w;
        pp = max(pp, curve_coeff[j]);
        aa[j] = 0;
        bb[j] = 0;
    }
    for (int j = 0; j < CURVES; j++)
    {
        curve_coeff[j] = exp(curve_coeff[j] - pp);
    }

    for (int i = 0; i < T; i++)
    {
        const int ii = i * C;
        const F kk = k[ii];
        const F vv = v[ii];

        F e1 = exp(u + kk);
        F e2 = exp(kk);

        F a = 0;
        F b = 0;

        for (int j = 0; j < CURVES; j++)
        {
            a += curve_coeff[j] * (aa[j] + e1 * vv);
            b += curve_coeff[j] * (bb[j] + e1);

            aa[j] = decay[j] * aa[j] + e2 * vv;
            bb[j] = decay[j] * bb[j] + e2;
        }

        y[ii] = a / b;
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C,
                                const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                                const F *__restrict__ const _y, const F *__restrict__ const _gy,
                                F *__restrict__ const _gw, F *__restrict__ const _gu, F *__restrict__ const _gk, F *__restrict__ const _gv)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const y = _y + _offset;
    const F *__restrict__ const gy = _gy + _offset;
    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F a[Tmax], b[Tmax];
    F log_evaluation_point[CURVES], decay[CURVES], curve_coeff[CURVES];
    F aa[CURVES], bb[CURVES];
    F sumv[CURVES], sumk[CURVES];
    F total_curve_coeff = 0, pp = MIN_VALUE;
    for (int j = 0; j < CURVES; j++)
    {
        F fraction = (F)j / (F)(CURVES - 1);
        log_evaluation_point[j] = (2 * fraction - 1) * log(CONTEXT_LENGTH);
        F evaluation_point = exp(log_evaluation_point[j]);
        decay[j] = exp(-1.0 / evaluation_point);
        curve_coeff[j] = -1.0 / evaluation_point + log_evaluation_point[j] * w;
        pp = max(pp, curve_coeff[j]);
        aa[j] = 0;
        bb[j] = 0;
        sumv[j] = 0;
        sumk[j] = 0;
    }
    for (int j = 0; j < CURVES; j++)
    {
        curve_coeff[j] = exp(curve_coeff[j] - pp);
        total_curve_coeff += curve_coeff[j];
    }

    F gw = 0, gu = 0, ga = 0, gb = 0;
    for (int i = 0; i < T; i++)
    {
        const int ii = i * C;
        const F kk = k[ii];
        const F vv = v[ii];
        const F yy = y[ii];

        F e1 = exp(u + kk);
        F e2 = exp(kk);

        a[i] = 0;
        b[i] = 0;

        for (int j = 0; j < CURVES; j++)
        {
            a[i] += curve_coeff[j] * (aa[j] + e1 * vv);
            b[i] += curve_coeff[j] * (bb[j] + e1);
        }

        gu += gy[ii] * total_curve_coeff * e1 * (vv - yy) / b[i];

        for (int j = 0; j < CURVES; j++)
        {
            gw += gy[ii] * curve_coeff[j] * log_evaluation_point[j] * (aa[j] + e1 * vv - yy * (bb[j] + e1)) / b[i];
            aa[j] = decay[j] * aa[j] + e2 * vv;
            bb[j] = decay[j] * bb[j] + e2;
        }
    }

    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] = gw * _w[_c]; // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = gu;

    for (int i = T - 1; i >= 0; i--)
    {
        const int ii = i * C;
        const F kk = k[ii];
        const F vv = v[ii];
        const F yy = y[ii];
        const F ek = exp(kk);
        const F eku = exp(kk + u);

        gk[ii] = total_curve_coeff * gy[ii] * eku * (vv - yy) / b[i];
        gv[ii] = total_curve_coeff * gy[ii] * eku / b[i];

        for (int j = 0; j < CURVES; j++)
        {
            gv[ii] += curve_coeff[j] * ek * sumv[j];
            gk[ii] += curve_coeff[j] * ek * (vv * sumv[j] - sumk[j]);
            sumv[j] = sumv[j] * decay[j] + gy[ii] / b[i];
            sumk[j] = sumk[j] * decay[j] + yy * gy[ii] / b[i];
        }
    }
}

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y)
{
    dim3 threadsPerBlock(min(C, 32)); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *gy, float *gw, float *gu, float *gk, float *gv)
{
    dim3 threadsPerBlock(min(C, 32)); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, gy, gw, gu, gk, gv);
}
