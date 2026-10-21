// crossbar_ir.c — 带线电阻 crossbar 的 block-SOR 求解内核（由 ir_drop/kernel.py 自动编译）
//
// 拓扑：G[i,j]，i = 行（WL，输入），j = 列（BL，输出）
//   WL i：驱动器 V_i —r— (i,0) —r— (i,1) … (i,N-1)（末端开路）
//   BL j：(0,j)（顶端开路）—r— … (N-1,j) —r— TIA 虚地（0 V）；输出 I_j = w[N-1,j]/r
// 布局：G, mwl, mbl : (N,N)；u, w, tmp : (N,N,K)；V : (N,K)
//
// ir_factor   : 预先分解每条 WL / BL 的三对角矩阵（Thomas 算法的消元系数）
// ir_sor_iter : 一次 block-SOR 迭代（先逐行精确求解 WL 节点 u，再逐列求解 BL 节点 w）；
//               out[0..2] = WL 节点最大变化、BL 节点最大变化、输出电流最大变化
#include <math.h>
#include <stddef.h>

void ir_factor(const double *G, long N, double r, double *mwl, double *mbl)
{
    const double gr = 1.0 / r, gr2 = gr * gr;
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        const double *g = G + i * N; double *m = mwl + i * N; double prev = 0.0;
        for (long j = 0; j < N; j++) {
            double diag = g[j] + gr + (j < N - 1 ? gr : 0.0);
            prev = 1.0 / (diag - (j > 0 ? gr2 * prev : 0.0));
            m[j] = prev;
        }
    }
    for (long j = 0; j < N; j++) mbl[j] = 1.0 / (G[j] + gr);
    for (long i = 1; i < N; i++) {
        const double *g = G + i * N, *mp = mbl + (i - 1) * N; double *m = mbl + i * N;
        #pragma omp parallel for schedule(static)
        for (long j = 0; j < N; j++) m[j] = 1.0 / (g[j] + 2.0 * gr - gr2 * mp[j]);
    }
}

void ir_sor_iter(const double *G, const double *mwl, const double *mbl, const double *V,
                 double *u, double *w, double *tmp, long N, long K, double r, double omega, double *out)
{
    const double gr = 1.0 / r;
    double du_max = 0.0, dw_max = 0.0, dI_max = 0.0;
    #pragma omp parallel for schedule(static) reduction(max:du_max)
    for (long i = 0; i < N; i++) {
        const double *g = G + i * N, *m = mwl + i * N;
        double *dp = tmp + i * N * K; const double *wr = w + i * N * K; double *ur = u + i * N * K;
        for (long k = 0; k < K; k++) dp[k] = (g[0] * wr[k] + V[i * K + k] * gr) * m[0];
        for (long j = 1; j < N; j++) {
            const double gj = g[j], mj = m[j]; const double *wj = wr + j * K, *dq = dp + (j - 1) * K;
            double *dj = dp + j * K;
            for (long k = 0; k < K; k++) dj[k] = (gj * wj[k] + dq[k] * gr) * mj;
        }
        double *xn = dp + (N - 1) * K;
        for (long k = 0; k < K; k++) {
            double *uu = ur + (N - 1) * K + k; double nu = (1.0 - omega) * (*uu) + omega * xn[k];
            double d = fabs(nu - *uu); if (d > du_max) du_max = d; *uu = nu;
        }
        for (long j = N - 2; j >= 0; j--) {
            const double c = m[j] * gr; double *dj = dp + j * K, *dn = dp + (j + 1) * K; double *uj = ur + j * K;
            for (long k = 0; k < K; k++) {
                double x = dj[k] + c * dn[k]; dj[k] = x;
                double nu = (1.0 - omega) * uj[k] + omega * x;
                double d = fabs(nu - uj[k]); if (d > du_max) du_max = d; uj[k] = nu;
            }
        }
    }
    for (long i = 0; i < N; i++) {
        const double *g = G + i * N, *m = mbl + i * N; const double *ui = u + i * N * K;
        double *di = tmp + i * N * K; const double *dq = (i > 0) ? tmp + (i - 1) * N * K : NULL;
        #pragma omp parallel for schedule(static)
        for (long j = 0; j < N; j++) {
            const double gj = g[j], mj = m[j];
            for (long k = 0; k < K; k++)
                di[j * K + k] = (gj * ui[j * K + k] + (dq ? dq[j * K + k] * gr : 0.0)) * mj;
        }
    }
    for (long i = N - 1; i >= 0; i--) {
        const double *m = mbl + i * N; double *di = tmp + i * N * K;
        const double *dn = (i < N - 1) ? tmp + (i + 1) * N * K : NULL; double *wi = w + i * N * K;
        double lw = 0.0, lI = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:lw, lI)
        for (long j = 0; j < N; j++) {
            const double c = m[j] * gr;
            for (long k = 0; k < K; k++) {
                long idx = j * K + k; double x = di[idx] + (dn ? c * dn[idx] : 0.0); di[idx] = x;
                double nw = (1.0 - omega) * wi[idx] + omega * x; double d = fabs(nw - wi[idx]);
                if (d > lw) lw = d; if (i == N - 1 && d * gr > lI) lI = d * gr; wi[idx] = nw;
            }
        }
        if (lw > dw_max) dw_max = lw; if (lI > dI_max) dI_max = lI;
    }
    out[0] = du_max; out[1] = dw_max; out[2] = dI_max;
}
