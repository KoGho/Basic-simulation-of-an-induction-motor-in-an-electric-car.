import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def voltageSynchronous(Vd, time, ma_t, f_t):
    omega = 2*np.pi*f_t
    # Syncrhonous voltages.
    Va = np.sqrt(2)*ma_t*Vd*np.sin(omega*time)
    Vb = np.sqrt(2)*ma_t*Vd*np.sin(omega*time+240/180*np.pi)
    Vc = np.sqrt(2)*ma_t*Vd*np.sin(omega*time+120/180*np.pi)
    return Va, Vb, Vc

  # Square wave voltages in a half bridge.
@nb.njit(parallel=True, fastmath=True)
def voltageInverter(time, steps, Vd, ma_t, f_t):
    omega = 2*np.pi*f_t
    v_ca = np.sqrt(2)*ma_t*np.sin(omega*time)
    v_cb = np.sqrt(2)*ma_t*np.sin(omega*time+240/180*np.pi)
    v_cc = np.sqrt(2)*ma_t*np.sin(omega*time+120/180*np.pi)

    v_s = np.zeros(steps)

    v_a = np.full(steps, np.nan)
    v_b = np.full(steps, np.nan)
    v_c = np.full(steps, np.nan)

    for l in range(steps-1):
        # Phase a.
        if v_ca[l] > v_s[l]:
            transistor_a = 1
        elif v_ca[l] <= v_s[l]:
            transistor_a = 0

        if transistor_a == 1:
            v_a[l] = Vd
        elif transistor_a == 0:
            v_a[l] = -Vd

        # Phase b.
        if v_cb[l] > v_s[l]:
            transistor_b = 1
        elif v_cb[l] <= v_s[l]:
            transistor_b = 0

        if transistor_b == 1:
            v_b[l] = Vd
        elif transistor_b == 0:
            v_b[l] = -Vd

        # Phase c.
        if v_cc[l] > v_s[l]:
            transistor_c = 1
        elif v_cc[l] <= v_s[l]:
            transistor_c = 0

        if transistor_c == 1:
            v_c[l] = Vd
        elif transistor_c == 0:
            v_c[l] = -Vd

    Va = v_a
    Vb = v_b
    Vc = v_c
    return Va, Vb, Vc

  # Pwm voltages.
@nb.njit(parallel=True, fastmath=True)
def bipolarPwm(m_a, mf, f, time, steps, Vd, ma_t, f_t):
    omega_s = mf*f*2*np.pi
    ts = 2*np.pi/omega_s
    omega = f_t*2*np.pi

    v_ca = np.sqrt(2)*ma_t*np.sin(omega*time)
    v_cb = np.sqrt(2)*ma_t*np.sin(omega*time+240/180*np.pi)
    v_cc = np.sqrt(2)*ma_t*np.sin(omega*time+120/180*np.pi)

    v_s = np.full(steps, np.nan)
    v_a = np.full(steps, np.nan)
    v_b = np.full(steps, np.nan)
    v_c = np.full(steps, np.nan)
    k = 0

    for l in range(steps-1):
        if time[l] >= k*ts and time[l] < k*ts + ts/4:
            v_s[l] = -4/ts*(time[l]-k*ts)
        elif time[l] >= k*ts + ts/4 and time[l] < k*ts+3*ts/4:
            v_s[l] = 4/ts*(time[l]-k*ts)-2
        elif time[l] >= k*ts + 3*ts/4 and time[l] < k*ts + ts:
            v_s[l] = -4/ts*(time[l]-k*ts)+4
        else:
            k = k+1

    for l in range(steps-1):
        # Phase a.
        if v_ca[l] > v_s[l]:
            transistor_a = 1
        elif v_ca[l] <= v_s[l]:
            transistor_a = 0

        if transistor_a == 1:
            v_a[l] = Vd
        elif transistor_a == 0:
            v_a[l] = -Vd

        # Phase b.
        if v_cb[l] > v_s[l]:
            transistor_b = 1
        elif v_cb[l] <= v_s[l]:
            transistor_b = 0

        if transistor_b == 1:
            v_b[l] = Vd
        elif transistor_b == 0:
            v_b[l] = -Vd

        # Phase c.
        if v_cc[l] > v_s[l]:
            transistor_c = 1
        elif v_cc[l] <= v_s[l]:
            transistor_c = 0

        if transistor_c == 1:
            v_c[l] = Vd
        elif transistor_c == 0:
            v_c[l] = -Vd

    Va = v_a
    Vb = v_b
    Vc = v_c
    return Va, Vb, Vc

  # Unipolar full bridge inverter pwm
@nb.njit(parallel=True, fastmath=True)
def unipolarPwm(m_a, mf, f, time, steps, Vd, ma_t, f_t):
    omega_s = mf*f*2*np.pi
    ts = 2*np.pi/omega_s
    omega = 2*np.pi*f_t

    v_caa = np.sqrt(2)*ma_t*np.sin(omega*time)
    v_cba = np.sqrt(2)*ma_t*np.sin(omega*time+240/180*np.pi)
    v_cca = np.sqrt(2)*ma_t*np.sin(omega*time+120/180*np.pi)

    v_cab = -np.sqrt(2)*ma_t*np.sin(omega*time)
    v_cbb = -np.sqrt(2)*ma_t*np.sin(omega*time+240/180*np.pi)
    v_ccb = -np.sqrt(2)*ma_t*np.sin(omega*time+120/180*np.pi)

    v_s = np.full(steps, np.nan)
    v_a = np.full(steps, np.nan)
    v_b = np.full(steps, np.nan)
    v_c = np.full(steps, np.nan)
    k = 0

    for l in range(steps-1):
        if time[l] >= k*ts and time[l] < k*ts + ts/4:
            v_s[l] = -4/ts*(time[l]-k*ts)
        elif time[l] >= k*ts + ts/4 and time[l] < k*ts+3*ts/4:
            v_s[l] = 4/ts*(time[l]-k*ts)-2
        elif time[l] >= k*ts + 3*ts/4 and time[l] < k*ts + ts:
            v_s[l] = -4/ts*(time[l]-k*ts)+4
        else:
            k = k+1

    for l in range(steps-1):
          # Phase a.
        if v_caa[l] > v_s[l]:
            transistor_aa = 1
        elif v_caa[l] <= v_s[l]:
            transistor_aa = 0

        if v_cab[l] > v_s[l]:
            transistor_ab = 1
        elif v_cab[l] <= v_s[l]:
            transistor_ab = 0

        if transistor_aa == 1 and transistor_ab == 0:
            v_a[l] = Vd
        elif transistor_aa == 0 and transistor_ab == 1:
            v_a[l] = -Vd
        elif transistor_aa == 1 and transistor_ab == 1:
            v_a[l] = 0
        elif transistor_aa == 0 and transistor_ab == 0:
            v_a[l] = 0

        # Phase b.
        if v_cba[l] > v_s[l]:
            transistor_ba = 1
        elif v_cba[l] <= v_s[l]:
            transistor_ba = 0

        if v_cbb[l] > v_s[l]:
            transistor_bb = 1
        elif v_cbb[l] <= v_s[l]:
            transistor_bb = 0

        if transistor_ba == 1 and transistor_bb == 0:
            v_b[l] = Vd
        elif transistor_ba == 0 and transistor_bb == 1:
            v_b[l] = -Vd
        elif transistor_ba == 1 and transistor_bb == 1:
            v_b[l] = 0
        elif transistor_ba == 0 and transistor_bb == 0:
            v_b[l] = 0

        # Phase c.
        if v_cca[l] > v_s[l]:
            transistor_ca = 1
        elif v_cca[l] <= v_s[l]:
            transistor_ca = 0

        if v_ccb[l] > v_s[l]:
            transistor_cb = 1
        elif v_ccb[l] <= v_s[l]:
            transistor_cb = 0

        if transistor_ca == 1 and transistor_cb == 0:
            v_c[l] = Vd
        elif transistor_ca == 0 and transistor_cb == 1:
            v_c[l] = -Vd
        elif transistor_ca == 1 and transistor_cb == 1:
            v_c[l] = 0
        elif transistor_ca == 0 and transistor_cb == 0:
            v_c[l] = 0

    Va = v_a
    Vb = v_b
    Vc = v_c
    return Va, Vb, Vc
