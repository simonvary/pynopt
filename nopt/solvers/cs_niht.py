from func.threshold_operators import *
from func.utils import *

import sys
from tqdm.auto import tqdm

def cs_niht(A, y, k, **kwargs):
    x_true = kwargs.get('x_true', None)
    x_init = kwargs.get('x_init', None)
    verb = kwargs.get('verb', False)
    tol_res = kwargs.get('tol_res', 1e-6)
    tol_rel = kwargs.get('tol_rel', 1-1e-5)
    rel_shift = kwargs.get('rel_shift', 15)
    max_iter = kwargs.get('max_iter', 100)

    error = {'res': np.zeros(max_iter), 'true': np.zeros(max_iter), 'rel': np.zeros(max_iter)}
    alphas = np.zeros(max_iter)

    if verb:
        print('Starting NIHT')
        pbar = tqdm(total = max_iter, file=sys.stdout)
        pbar.update(1)

    w = A.backward(y)
    T_k, x = thresh_hard_sparse(w, k)
    if x_true is not None:
        error['true'][0] = relerror(x, x_true)
    error['res'][0]  = relerror(A.forward(x), y)
    error['rel'][0]  = 0
    alphas[0] = 1
    # Iterative process
    l = 1
    not_finished = True
    while not_finished:
        r = A.backward(y-A.forward(x))
        r_proj = support_projection(r, T_k)
        a = np.linalg.norm(r_proj)**2
        b = np.linalg.norm(A.forward(r_proj))**2
        alpha = a/b
        w = x + alpha * r
        T_k, x = thresh_hard_sparse(w, k)
        alphas[l] = alpha

        # Error tracking:
        error['res'][l] = relerror(A.forward(x), y)
        if x_true is not None:
            error['true'][l] = relerror(x, x_true)
        if l >= rel_shift:
            error['rel'][l] = (error['res'][l]/error['res'][l-rel_shift])**(1/rel_shift)
        else:
            error['rel'][l] = 0

        if verb:
            pbar.set_postfix(error_res = error['res'][l],
                            error_true = error['true'][l],
                            error_rel  = error['rel'][l])
            pbar.update(1)

        not_finished = (l < max_iter - 1) and (error['res'][l] > tol_res) and (error['rel'][l] < tol_rel)
        l = l + 1

    # Truncate tracker variables
    for item in error:
        error[item] = error[item][:l]
    alphas = alphas[:l]

    out = {'alphas': alphas}
    return (x, error, out)