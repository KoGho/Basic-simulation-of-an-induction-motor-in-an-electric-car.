import numpy as np
import matplotlib.pyplot as plt
from machine_model import machineModel
from park import inversePark
from park import parkF
from voltages import voltageSynchronous
from voltages import voltageInverter
from voltages import bipolarPwm
from voltages import unipolarPwm
from ramps import start
from ramps import stop
from ramps import during
import numba as nb
from utilities import plots
from utilities import computeVoltage
from utilities import selectVoltage
from auto_model import autoModel

class InductionMachine:
    def __init__(self):
        # Fixed parameters of the machine
        #self.R_s = 240e-3
        #self.L_s = 59.4e-3
        #self.R_r = 175e-3
        #self.L_r = 59.1e-3
        #self.L_m = 57e-3
        self.R_s = 12e-3#125
        self.L_s = 55.9e-3#57.9
        self.R_r = 20e-3#125
        self.L_r = 56.8e-3#57.6
        self.L_m = 56.3e-3#57.5
        self.n = 2#3
        self.J = 0.4#0.7
        self.k_0 = 9#100
        self.k_1 = 0.025#0.1
        self.k_0a = 1200
        self.k_1a = 0.08
        self.k_0b = -1200
        self.gear = 10
        self.k_1b = -0.08
        self.Vd = 400/np.sqrt(3)#1500
        self.mf = 20#40
        # Simulation parameters
        self.t = 5e-7
        self.T = 1
        self.steps = int(1/5e-7)
        self.time = np.arange(0,1,5e-7)
        self.time2 = np.arange(1,2,5e-7)
        self.time3 = np.arange(2,3,5e-7)
        self.time4 = np.arange(3,4,5e-7)
        self.time5 = np.arange(4,5,5e-7)
        self.time6 = np.arange(5,6,5e-7)
        # Variables that can be changed
        self.ma = 1
        self.f = 80
        self.omega = 2*np.pi*80
        self.maximum = 550
        # Variables to start the motor
        self.ramp_s = 1
        self.ramp_f = 0.5
        # To control the output of the machine I should change the values of ma, Vd
        # mf, f, omega

    # Run simulation.
    def runSim(self):
        what = '1'
        while what == '1' or what == '2':
            print('Do you want to simulate the speed-climb of the car')
            print('or the car going up and down a small hill?')
            print('1 - Speed-climb')
            print('2 - Small-hill')
            print('q - Quit')
            what = input('Enter your choice (1/2):')
            if what == '1':
                print('0-100 clib or 0-200 climb?')
                print('1 - 0-100 climb')
                print('2 - 0-200 climb')
                climb = input('Enter your choice (1/2):')
                if climb == '1':
                    self.f = 270
                    self.omega = 2*np.pi*270
                    self.T = 6
                    self.steps = int(6/self.t)
                    self.time = np.arange(0,6,self.t)
                    self.ramp_s = 5.2
                elif climb == '2':
                    self.T = 13
                    self.steps = int(13/self.t)
                    self.time = np.arange(0,13,self.t)
                    self.f = 270
                    self.omega = 2*np.pi*270
                    self.ramp_s = 5.2
                else:
                    raise ValueError('Invalid input.')
                # Compute the starting ramps.
                (ma_t, F_t) = start(self.ma, self.steps, self.t, self.f, self.ramp_s, self.maximum)

                # Compute the source voltages.
                (V_s, value) = selectVoltage(ma_t, F_t, self.time, self.Vd, self.steps, self.ma, self.mf, self.f)

                (omega_mt, Omega_mt, psi_st, psi_rt, i_st, i_rt, theta_t, \
                    theta_rt, c_et, c_mt, i_at, i_bt, i_ct, i_art, i_brt, i_crt, \
                    psi_at, psi_bt, psi_ct, psi_art, psi_brt, psi_crt, v_rt) = machineModel(V_s, \
                self.steps, self.n, self.t, self.J, self.R_s, F_t, self.R_r, \
                self.L_r, self.L_m, self.L_s, self.k_0, self.k_1, \
                0, 0, 0, 0, self.k_0, 0, 0, 0, 0, 0, \
                0, 0, 0, -np.pi/2, 0, 0, 0, 0, 0, \
                0, 0, 0, V_s[0], F_t[0], self.gear)
                total_t = self.time
            elif what == '2':
                print('Do you want to use cruise control?')
                print('1 - Yes')
                print('2 - No')
                cruise = input('Enter your choice (1/2): ')

                if cruise == '1':
                    cruise_control = 1
                elif cruise == '2':
                    cruise_control = 2
                else:
                    raise ValueError('Invalid input')

                # Automatic or not second path
                (ma_t, F1) = start(self.ma, self.steps, self.t, self.f, self.ramp_s, self.maximum)

                # Compute the source voltages.
                (V_s1, value) = selectVoltage(ma_t, F1, self.time, self.Vd, self.steps, self.ma, self.mf, self.f)


                (omega_m1, Omega_m1, psi_s1, psi_r1, i_s1, i_r1, theta1, \
                    theta_r1, c_e1, c_m1, i_a1, i_b1, i_c1, i_ar1, i_br1, i_cr1, \
                    psi_a1, psi_b1, psi_c1, psi_ar1, psi_br1, psi_cr1, v_r1) = machineModel(V_s1, \
                self.steps, self.n, self.t, self.J, self.R_s, F1, self.R_r, \
                self.L_r, self.L_m, self.L_s, self.k_0, self.k_1, \
                0, 0, 0, 0, self.k_0, 0, 0, 0, 0, 0, \
                0, 0, 0, -np.pi/2, 0, 0, 0, 0, 0, \
                0, 0, 0, V_s1[0], F1[0], self.gear)

                end = self.steps-1
                v_s1 = V_s1[end]
                f1 = F1[end]

                if cruise_control == 1:

                    (omega_m2, Omega_m2, psi_s2, psi_r2, i_s2, i_r2, theta2, \
                        theta_r2, c_e2, c_m2, i_a2, i_b2, i_c2, i_ar2, i_br2, i_cr2, \
                        psi_a2, psi_b2, psi_c2, psi_ar2, psi_br2, psi_cr2, v_r2, V_s2, F2) = autoModel( \
                    self.steps, self.n, self.t, self.J, self.R_s, self.R_r, \
                    self.L_r, self.L_m, self.L_s, self.k_0a, self.k_1a, \
                    psi_s1[end], psi_r1[end], omega_m1[end], c_e1[end], c_m1[end], i_s1[end], i_r1[end], i_a1[end], i_b1[end], i_c1[end], \
                    psi_a1[end], psi_b1[end], psi_c1[end], theta1[end], i_ar1[end], i_br1[end], i_cr1[end], psi_ar1[end], psi_br1[end], \
                    psi_cr1[end], theta_r1[end], v_r1[end], value, self.Vd, self.ma, self.f, self.mf, v_s1, f1, self.gear)
                    v_s2 = V_s2[end]
                    f2 = F2[end]

                    (omega_m3, Omega_m3, psi_s3, psi_r3, i_s3, i_r3, theta3, \
                        theta_r3, c_e3, c_m3, i_a3, i_b3, i_c3, i_ar3, i_br3, i_cr3, \
                        psi_a3, psi_b3, psi_c3, psi_ar3, psi_br3, psi_cr3, v_r3, V_s3, F3) = autoModel( \
                    self.steps, self.n, self.t, self.J, self.R_s, self.R_r, \
                    self.L_r, self.L_m, self.L_s, self.k_0, self.k_1, \
                    psi_s2[end], psi_r2[end], omega_m2[end], c_e2[end], c_m2[end], i_s2[end], i_r2[end], i_a2[end], i_b2[end], i_c2[end], \
                    psi_a2[end], psi_b2[end], psi_c2[end], theta2[end], i_ar2[end], i_br2[end], i_cr2[end], psi_ar2[end], psi_br2[end], \
                    psi_cr2[end], theta_r2[end], v_r2[end], value, self.Vd, self.ma, self.f, self.mf, v_s2, f2, self.gear)
                    v_s3 = V_s3[end]
                    f3 = F3[end]

                    (omega_m4, Omega_m4, psi_s4, psi_r4, i_s4, i_r4, theta4, \
                        theta_r4, c_e4, c_m4, i_a4, i_b4, i_c4, i_ar4, i_br4, i_cr4, \
                        psi_a4, psi_b4, psi_c4, psi_ar4, psi_br4, psi_cr4, v_r4, V_s4, F4) = autoModel( \
                    self.steps, self.n, self.t, self.J, self.R_s, self.R_r, \
                    self.L_r, self.L_m, self.L_s, self.k_0b, self.k_1b, \
                    psi_s3[end], psi_r3[end], omega_m3[end], c_e3[end], c_m3[end], i_s3[end], i_r3[end], i_a3[end], i_b3[end], i_c3[end], \
                    psi_a3[end], psi_b3[end], psi_c3[end], theta3[end], i_ar3[end], i_br3[end], i_cr3[end], psi_ar3[end], psi_br3[end], \
                    psi_cr3[end], theta_r3[end], v_r3[end], value, self.Vd, self.ma, self.f, self.mf, v_s3, f3, self.gear)
                    v_s4 = V_s4[end]
                    f4 = F4[end]

                    (omega_m5, Omega_m5, psi_s5, psi_r5, i_s5, i_r5, theta5, \
                        theta_r5, c_e5, c_m5, i_a5, i_b5, i_c5, i_ar5, i_br5, i_cr5, \
                        psi_a5, psi_b5, psi_c5, psi_ar5, psi_br5, psi_cr5, v_r5, V_s5, F5) = autoModel( \
                    self.steps, self.n, self.t, self.J, self.R_s, self.R_r, \
                    self.L_r, self.L_m, self.L_s, self.k_0, self.k_1, \
                    psi_s4[end], psi_r4[end], omega_m4[end], c_e4[end], c_m4[end], i_s4[end], i_r4[end], i_a4[end], i_b4[end], i_c4[end], \
                    psi_a4[end], psi_b4[end], psi_c4[end], theta4[end], i_ar4[end], i_br4[end], i_cr4[end], psi_ar4[end], psi_br4[end], \
                    psi_cr4[end], theta_r4[end], v_r4[end], value, self.Vd, self.ma, self.f, self.mf, v_s4, f4, self.gear)
                    v_s5 = V_s5[end]
                    f5 = F5[end]
                elif cruise_control == 2:
                    # Compute the transient after an increase in the value of the load. (uphill)
                    (ma_t, F2) = during(self.ma, self.steps, self.t, self.f)

                    # Compute the source voltages.
                    (V_s2) = computeVoltage(ma_t, F2, self.time2, value, self.Vd, self.steps, self.ma, self.mf, self.f)

                    (omega_m2, Omega_m2, psi_s2, psi_r2, i_s2, i_r2, theta2, \
                        theta_r2, c_e2, c_m2, i_a2, i_b2, i_c2, i_ar2, i_br2, i_cr2, \
                        psi_a2, psi_b2, psi_c2, psi_ar2, psi_br2, psi_cr2, v_r2) = machineModel(V_s2, \
                    self.steps, self.n, self.t, self.J, self.R_s, F2, self.R_r, \
                    self.L_r, self.L_m, self.L_s, self.k_0a, self.k_1a, \
                    psi_s1[end], psi_r1[end], omega_m1[end], c_e1[end], c_m1[end], i_s1[end], i_r1[end], i_a1[end], i_b1[end], i_c1[end], \
                    psi_a1[end], psi_b1[end], psi_c1[end], theta1[end], i_ar1[end], i_br1[end], i_cr1[end], psi_ar1[end], psi_br1[end], \
                    psi_cr1[end], theta_r1[end], v_r1[end], v_s1, f1, self.gear)
                    v_s2 = V_s2[end]
                    f2 = F2[end]

                    # Compute the transient after a decrease in the value of the load. (level)
                    (ma_t, F3) = during(self.ma, self.steps, self.t, self.f)

                    # Compute the source voltages.
                    (V_s3) = computeVoltage(ma_t, F3, self.time3, value, self.Vd, self.steps, self.ma, self.mf, self.f)

                    (omega_m3, Omega_m3, psi_s3, psi_r3, i_s3, i_r3, theta3, \
                        theta_r3, c_e3, c_m3, i_a3, i_b3, i_c3, i_ar3, i_br3, i_cr3, \
                        psi_a3, psi_b3, psi_c3, psi_ar3, psi_br3, psi_cr3, v_r3) = machineModel(V_s3, \
                    self.steps, self.n, self.t, self.J, self.R_s, F3, self.R_r, \
                    self.L_r, self.L_m, self.L_s, self.k_0, self.k_1, \
                    psi_s2[end], psi_r2[end], omega_m2[end], c_e2[end], c_m2[end], i_s2[end], i_r2[end], i_a2[end], i_b2[end], i_c2[end], \
                    psi_a2[end], psi_b2[end], psi_c2[end], theta2[end], i_ar2[end], i_br2[end], i_cr2[end], psi_ar2[end], psi_br2[end], \
                    psi_cr2[end], theta_r2[end], v_r2[end], v_s2, f2, self.gear)
                    v_s3 = V_s3[end]
                    f3 = F3[end]

                    # Compute the transient after a negative load. (downhill)
                    (ma_t, F4) = during(self.ma, self.steps, self.t, self.f)

                    # Compute the source voltages.
                    (V_s4) = computeVoltage(ma_t, F4, self.time4, value, self.Vd, self.steps, self.ma, self.mf, self.f)

                    (omega_m4, Omega_m4, psi_s4, psi_r4, i_s4, i_r4, theta4, \
                        theta_r4, c_e4, c_m4, i_a4, i_b4, i_c4, i_ar4, i_br4, i_cr4, \
                        psi_a4, psi_b4, psi_c4, psi_ar4, psi_br4, psi_cr4, v_r4) = machineModel(V_s4, \
                    self.steps, self.n, self.t, self.J, self.R_s, F4, self.R_r, \
                    self.L_r, self.L_m, self.L_s, self.k_0b, self.k_1b, \
                    psi_s3[end], psi_r3[end], omega_m3[end], c_e3[end], c_m3[end], i_s3[end], i_r3[end], i_a3[end], i_b3[end], i_c3[end], \
                    psi_a3[end], psi_b3[end], psi_c3[end], theta3[end], i_ar3[end], i_br3[end], i_cr3[end], psi_ar3[end], psi_br3[end], \
                    psi_cr3[end], theta_r3[end], v_r3[end], v_s3, f3, self.gear)
                    v_s4 = V_s4[end]
                    f4 = F4[end]

                    # Compute the transient after a negative load. (downhill)
                    (ma_t, F5) = during(self.ma, self.steps, self.t, self.f)

                    # Compute the source voltages.
                    (V_s5) = computeVoltage(ma_t, F5, self.time6, value, self.Vd, self.steps, self.ma, self.mf, self.f)

                    (omega_m5, Omega_m5, psi_s5, psi_r5, i_s5, i_r5, theta5, \
                        theta_r5, c_e5, c_m5, i_a5, i_b5, i_c5, i_ar5, i_br5, i_cr5, \
                        psi_a5, psi_b5, psi_c5, psi_ar5, psi_br5, psi_cr5, v_r5) = machineModel(V_s5, \
                    self.steps, self.n, self.t, self.J, self.R_s, F5, self.R_r, \
                    self.L_r, self.L_m, self.L_s, self.k_0, self.k_1, \
                    psi_s4[end], psi_r4[end], omega_m4[end], c_e4[end], c_m4[end], i_s4[end], i_r4[end], i_a4[end], i_b4[end], i_c4[end], \
                    psi_a4[end], psi_b4[end], psi_c4[end], theta4[end], i_ar4[end], i_br4[end], i_cr4[end], psi_ar4[end], psi_br4[end], \
                    psi_cr4[end], theta_r4[end], v_r4[end], v_s4, f4, self.gear)
                    v_s5 = ma_t[end]*np.sqrt(2)*self.Vd
                    f5 = F5[end]
                else:
                    raise ValueError('An error occured.')

                # Compute the stopping ramp.
                (ma_t, F6) = stop(np.real(v_s5)/(np.sqrt(2)*self.Vd), self.steps, self.t, f5, self.ramp_f)

                # Compute the source voltages.
                (V_s6) = computeVoltage(ma_t, F6, self.time, value, self.Vd, self.steps, self.ma, self.mf, self.f)

                (omega_m6, Omega_m6, psi_s6, psi_r6, i_s6, i_r6, theta6, \
                    theta_r6, c_e6, c_m6, i_a6, i_b6, i_c6, i_ar6, i_br6, i_cr6, \
                    psi_a6, psi_b6, psi_c6, psi_ar6, psi_br6, psi_cr6, v_r6) = machineModel(V_s6, \
                self.steps, self.n, self.t, self.J, self.R_s, F6, self.R_r, \
                self.L_r, self.L_m, self.L_s, self.k_0, self.k_1, \
                psi_s5[end], psi_r5[end], omega_m5[end], c_e5[end], c_m5[end], i_s5[end], i_r5[end], i_a5[end], i_b5[end], i_c5[end], \
                psi_a5[end], psi_b5[end], psi_c5[end], -np.pi/2, i_ar5[end], i_br5[end], i_cr5[end], psi_ar5[end], psi_br5[end], \
                psi_cr5[end], theta_r5[end], v_r5[end], v_s5, f5, self.gear)

                total_t = np.concatenate((self.time, self.time2, self.time3, self.time4, self.time5, self.time6))

                Omega_mt = np.concatenate((Omega_m1, Omega_m2, Omega_m3, Omega_m4, Omega_m5, Omega_m6))
                omega_mt = np.concatenate((omega_m1, omega_m2, omega_m3, omega_m4, omega_m5, omega_m6))

                i_at = np.concatenate((i_a1, i_a2, i_a3, i_a4, i_a5, i_a6))
                i_bt = np.concatenate((i_b1, i_b2, i_b3, i_b4, i_b5, i_b6))
                i_ct = np.concatenate((i_c1, i_c2, i_c3, i_c4, i_c5, i_c6))

                i_art = np.concatenate((i_ar1, i_ar2, i_ar3, i_ar4, i_ar5, i_ar6))
                i_brt = np.concatenate((i_br1, i_br2, i_br3, i_br4, i_br5, i_br6))
                i_crt = np.concatenate((i_cr1, i_cr2, i_cr3, i_cr4, i_cr5, i_cr6))

                psi_at = np.concatenate((psi_a1, psi_a2, psi_a3, psi_a4, psi_a5, psi_a6))
                psi_bt = np.concatenate((psi_b1, psi_b2, psi_b3, psi_b4, psi_b5, psi_b6))
                psi_ct = np.concatenate((psi_c1, psi_c2, psi_c3, psi_c4, psi_c5, psi_c6))

                psi_art = np.concatenate((psi_ar1, psi_ar2, psi_ar3, psi_ar4, psi_ar5, psi_ar6))
                psi_brt = np.concatenate((psi_br1, psi_br2, psi_br3, psi_br4, psi_br5, psi_br6))
                psi_crt = np.concatenate((psi_cr1, psi_cr2, psi_cr3, psi_cr4, psi_cr5, psi_cr6))

                psi_st = np.concatenate((psi_s1, psi_s2, psi_s3, psi_s4, psi_s5, psi_s6))
                psi_rt = np.concatenate((psi_r1, psi_r2, psi_r3, psi_r4, psi_r5, psi_r6))

                i_st = np.concatenate((i_s1, i_s2, i_s3, i_s4, i_s5, i_s6))
                i_rt = np.concatenate((i_r1, i_r2, i_r3, i_r4, i_r5, i_r6))

                theta_t = np.concatenate((theta1, theta2, theta3, theta4, theta5, theta6))
                theta_rt = np.concatenate((theta_r1, theta_r2, theta_r3, theta_r4, theta_r5, theta_r6))

                c_et = np.concatenate((c_e1, c_e2, c_e3, c_e4, c_e5, c_e6))
                c_mt = np.concatenate((c_m1, c_m2, c_m3, c_m4, c_m5, c_m6))
                V_s = np.concatenate((V_s1, V_s2, V_s3, V_s4, V_s5, V_s6))
                F_t = np.concatenate((F1, F2, F3, F4, F5, F6))
            elif what == 'q':
                print('Simulation stopped')
                break
            else:
                raise ValueError('Invalid input.')

            # Display images.
            plots(omega_mt, Omega_mt, psi_st, psi_rt, i_st, i_rt, theta_t, \
                theta_rt, c_et, c_mt, i_at, i_bt, i_ct, i_art, i_brt, i_crt, \
                psi_at, psi_bt, psi_ct, psi_art, psi_brt, psi_crt, total_t, \
                self.R_r, self.R_s, self.L_r, self.L_s, self.L_m, V_s, F_t)


    def changeParameter(self):
            print('Select parameter to change:')
            print('1 - ma')
            print('2 - f')
            choice = input('Enter your choice (1/2): ')
            match choice:
                case '1':
                    print('Starting from 1 select:')
                    print('1 to reduce ma by 0.1')
                    print('2 to increase ma by 0.1')
                    other = input('Enter your choice (1/2): ')
                    match other:
                        case '1':
                            self.ma = self.ma-0.1
                        case '2':
                            self.ma = self.ma+0.1
                        case _:
                            raise ValueError('Invalid choice. Please select 1 or 2.')
                case '2':
                    print('Starting from 50 Hz select:')
                    print('1 to reduce f by 10 Hz')
                    print('2 to increase f by 10 Hz')
                    other = input('Enter your choice (1/2): ')
                    match other:
                        case '1':
                            self.f = self.f-10
                            self.omega = self.omega - 2*np.pi*10
                        case '2':
                            self.f = self.f+10
                            self.omega = self.omega + 2*np.pi*10
                        case _:
                            raise ValueError('Invalid choice. Please select 1 or 2.')
                case _:
                    raise ValueError('Invalid choice. Please select 1 or 2.')
