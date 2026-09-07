# D-SCC plan: v4.5 amendment — SCF initialisation during training (apply to §2.6, §5)
- First visit to a frame in an arm: continuation from Φ = 0 in the registered four stages
  with tangent predictor; converged `dq` stored per frame id.
- Later visits: warm start from the stored `dq`; Newton with Levenberg–Marquardt damping
  (no trial re-diagonalisations); converge to the registered `tol_q`, `tol_E`, `tol_c`.
- Registered per-epoch subsample (default 5 % of frames): full continuation re-run and
  compared with the warm-started solution; `|dq_warm - dq_cont| < tol_root` required;
  the failing fraction is logged and subject to the registered ceiling; failure fails
  the arm. Physics, fixed points, tolerances and gradients unchanged.
- Solver engineering permitted without further ruling: batched Jacobian/residual
  einsum, CUDA graphs/compile, eigensolver backend swap, float32 pre-iterations with
  float64 final convergence. Gate before restart: fixed points, band identity, FD forces
  and implicit gradients agree with the previous implementation to tolerance.
