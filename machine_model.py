import numpy as np
import numba as nb
from park import inversePark

# gear in the function

@nb.njit(parallel=True, fastmath=True)
def machineModel(v_s, steps, n, t, J, R_s, f_t, R_r, L_r, L_m, L_s, k_0, k_1, \
        psi_s0, psi_r0, omega_m0, c_e0, c_m0, i_s0, i_r0, i_a0, i_b0, i_c0, \
        psi_a0, psi_b0, psi_c0, theta0, i_ar0, i_br0, i_cr0, psi_ar0, psi_br0, \
        psi_cr0, theta_r0, v_r0, v_s0, f0, gear):
    # Initialize arrays
    f_t[0] = f_t[1] = f0
    omega = 2*np.pi*f_t
    v_s[0] = v_s[1] = v_s0
    psi_s = np.zeros(steps, dtype=np.complex128)
    psi_r = np.zeros(steps, dtype=np.complex128)
    omega_m = np.zeros(steps)
    Omega_m = omega_m / n
    c_e = np.zeros(steps)
    c_m = np.zeros(steps)
    i_s = np.zeros(steps, dtype=np.complex128)
    i_r = np.zeros(steps, dtype=np.complex128)

    i_a = np.zeros(steps)
    i_b = np.zeros(steps)
    i_c = np.zeros(steps)
    psi_a = np.zeros(steps)
    psi_b = np.zeros(steps)
    psi_c = np.zeros(steps)
    theta = np.zeros(steps)
    i_ar = np.zeros(steps)
    i_br = np.zeros(steps)
    i_cr = np.zeros(steps)
    psi_ar = np.zeros(steps)
    psi_br = np.zeros(steps)
    psi_cr = np.zeros(steps)
    theta_r = np.zeros(steps)
    v_r = np.zeros(steps)

    # Assign initial conditions.
    psi_s[0] = psi_s[1] = psi_s0
    psi_r[0] = psi_r[1] = psi_r0
    omega_m[0] = omega_m[1] = omega_m0
    Omega_m[0] = Omega_m[1] = omega_m0 / n
    c_e[0] = c_e[1] = c_e0
    c_m[0] = c_m[1] = c_m0
    i_s[0] = i_s[1] = i_s0
    i_r[0] = i_r[1] = i_r0

    i_a[0] = i_a[1] = i_a0
    i_b[0] = i_b[1] = i_b0
    i_c[0] = i_c[1] = i_c0
    psi_a[0] = psi_a[1] = psi_a0
    psi_b[0] = psi_b[1] = psi_b0
    psi_c[0] = psi_c[1] = psi_c0
    theta[0] = theta[1] = theta0
    i_ar[0] = i_ar[1] = i_ar0
    i_br[0] = i_br[1] = i_br0
    i_cr[0] = i_cr[1] = i_cr0
    psi_ar[0] = psi_ar[1] = psi_ar0
    psi_br[0] = psi_br[1] = psi_br0
    psi_cr[0] = psi_cr[1] = psi_cr0
    theta_r[0] = theta_r[1] = theta_r0
    v_r[0] = v_r[1] = v_r0

    # Main loop
    for k in range(steps-1):
        omega_m[k+1] = omega_m[k] + t * ((n/J) * (c_e[k] - c_m[k]))
        Omega_m[k+1] = omega_m[k+1] / n
        psi_s[k+1] = psi_s[k] + t * (v_s[k] - R_s * i_s[k] - 1j * omega[k] * \
            psi_s[k])
        psi_r[k+1] = psi_r[k] + t * (v_r[k] - R_r * i_r[k] - 1j * psi_r[k] * \
            (omega[k] - omega_m[k]))
        i_s[k+1] = (L_r * psi_s[k+1] - L_m * psi_r[k+1]) / (L_s * L_r - L_m**2)
        i_r[k+1] = (L_s * psi_r[k+1] - L_m * psi_s[k+1]) / (L_s * L_r - L_m**2)
        theta[k+1] = theta[k] + t * omega[k]
        theta_r[k+1] = theta_r[k] + t * (omega[k] - omega_m[k])
        c_e[k+1] = n * np.imag(i_s[k+1] * np.conj(psi_s[k+1]))
        c_tire = k_0 + k_1*Omega_m[k+1]/gear + k_1 * (Omega_m[k+1]/gear)**2
        c_m[k+1] = c_tire/gear

        (i_a[k], i_b[k], i_c[k]) = inversePark(np.real(i_s[k]), \
            (np.imag(i_s[k])), theta[k])
        (i_ar[k], i_br[k], i_cr[k]) = inversePark(np.real(i_r[k]), \
            (np.imag(i_r[k])), theta_r[k])
        (psi_a[k], psi_b[k], psi_c[k]) = inversePark(np.real(psi_s[k]), \
            (np.imag(psi_s[k])), theta[k])
        (psi_ar[k], psi_br[k], psi_cr[k]) = inversePark(np.real(psi_r[k]), \
            (np.imag(psi_r[k])), theta_r[k])

    return omega_m, Omega_m, psi_s, psi_r, i_s, i_r, theta, \
           theta_r, c_e, c_m, i_a, i_b, i_c, i_ar, i_br, i_cr, \
           psi_a, psi_b, psi_c, psi_ar, psi_br, psi_cr, v_r
