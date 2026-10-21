// crossbar_ir_rect.c — block-SOR kernel for RECTANGULAR crossbars with wire resistance.
// Compiled automatically by ir_drop/kernel.py (load_rect_kernel). The square kernel in
// crossbar_ir.c is left untouched so that square-array results stay bit-identical.
//
// Topology (identical to crossbar_ir.c, with R rows and C columns):
//   G[i,j], i = row (word line, input), j = column (bit line, output)
//   WL i: driver V_i -r- (i,0) -r- (i,1) ... (i,C-1)   (far end open)
//   BL j: (0,j) (top end open) -r- ... (R-1,j) -r- TIA virtual ground (0 V); I_j = w[R-1,j] / r
// Layout: G, mwl, mbl : (R,C);  u, w, tmp : (R,C,K);  V : (R,K)
//
// ir_factor_rect   : Thomas-algorithm elimination factors of every WL (length C) and BL (length R)
// ir_sor_iter_rect : one block-SOR sweep (exact WL solves for u, then exact BL solves for w);
//                    out[0..2] = max change of WL nodes, BL nodes and output currents
#include <math.h>
#include <stddef.h>

void ir_factor_rect(const double *G, long R, long C, double r, double *mwl, double *mbl)
{
    const double gr = 1.0 / r, gr2 = gr * gr;
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < R; i++) {
        const double *g = G + i * C; double *m = mwl + i * C; double prev = 0.0;
        for (long j = 0; j < C; j++) {
            double diag = g[j] + gr + (j < C - 1 ? gr : 0.0);
            prev = 1.0 / (diag - (j > 0 ? gr2 * prev : 0.0));
            m[j] = prev;
        }
    }
    for (long j = 0; j < C; j++) mbl[j] = 1.0 / (G[j] + gr);
    for (long i = 1; i < R; i++) {
        const double *g = G + i * C, *mp = mbl + (i - 1) * C; double *m = mbl + i * C;
        #pragma omp parallel for schedule(static)
        for (long j = 0; j < C; j++) m[j] = 1.0 / (g[j] + 2.0 * gr - gr2 * mp[j]);
    }
}

void ir_sor_iter_rect(const double *G, const double *mwl, const double *mbl, const double *V,
                      double *u, double *w, double *tmp, long R, long C, long K, double r, double omega,
                      double *out)
{
    const double gr = 1.0 / r;
    double du_max = 0.0, dw_max = 0.0, dI_max = 0.0;
    // ---- WL sweep: rows are independent ----
    #pragma omp parallel for schedule(static) reduction(max:du_max)
    for (long i = 0; i < R; i++) {
        const double *g = G + i * C, *m = mwl + i * C;
        double *dp = tmp + i * C * K; const double *wr = w + i * C * K; double *ur = u + i * C * K;
        for (long k = 0; k < K; k++) dp[k] = (g[0] * wr[k] + V[i * K + k] * gr) * m[0];
        for (long j = 1; j < C; j++) {
            const double gj = g[j], mj = m[j]; const double *wj = wr + j * K, *dq = dp + (j - 1) * K;
            double *dj = dp + j * K;
            for (long k = 0; k < K; k++) dj[k] = (gj * wj[k] + dq[k] * gr) * mj;
        }
        double *xn = dp + (C - 1) * K;
        for (long k = 0; k < K; k++) {
            double *uu = ur + (C - 1) * K + k; double nu = (1.0 - omega) * (*uu) + omega * xn[k];
            double d = fabs(nu - *uu); if (d > du_max) du_max = d; *uu = nu;
        }
        for (long j = C - 2; j >= 0; j--) {
            const double c = m[j] * gr; double *dj = dp + j * K, *dn = dp + (j + 1) * K; double *uj = ur + j * K;
            for (long k = 0; k < K; k++) {
                double x = dj[k] + c * dn[k]; dj[k] = x;
                double nu = (1.0 - omega) * uj[k] + omega * x;
                double d = fabs(nu - uj[k]); if (d > du_max) du_max = d; uj[k] = nu;
            }
        }
    }
    // ---- BL sweep: forward elimination down the rows ----
    for (long i = 0; i < R; i++) {
        const double *g = G + i * C, *m = mbl + i * C; const double *ui = u + i * C * K;
        double *di = tmp + i * C * K; const double *dq = (i > 0) ? tmp + (i - 1) * C * K : NULL;
        #pragma omp parallel for schedule(static)
        for (long j = 0; j < C; j++) {
            const double gj = g[j], mj = m[j];
            for (long k = 0; k < K; k++)
                di[j * K + k] = (gj * ui[j * K + k] + (dq ? dq[j * K + k] * gr : 0.0)) * mj;
        }
    }
    // ---- BL sweep: back substitution up the rows ----
    for (long i = R - 1; i >= 0; i--) {
        const double *m = mbl + i * C; double *di = tmp + i * C * K;
        const double *dn = (i < R - 1) ? tmp + (i + 1) * C * K : NULL; double *wi = w + i * C * K;
        double lw = 0.0, lI = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:lw, lI)
        for (long j = 0; j < C; j++) {
            const double c = m[j] * gr;
            for (long k = 0; k < K; k++) {
                long idx = j * K + k; double x = di[idx] + (dn ? c * dn[idx] : 0.0); di[idx] = x;
                double nw = (1.0 - omega) * wi[idx] + omega * x; double d = fabs(nw - wi[idx]);
                if (d > lw) lw = d; if (i == R - 1 && d * gr > lI) lI = d * gr; wi[idx] = nw;
            }
        }
        if (lw > dw_max) dw_max = lw; if (lI > dI_max) dI_max = lI;
    }
    out[0] = du_max; out[1] = dw_max; out[2] = dI_max;
}
