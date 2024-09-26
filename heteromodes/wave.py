import numpy as np
from scipy.integrate import odeint
from heteromodes.eigentools import calc_eigendecomposition

# ================================
# ========== WAVE MODEL ==========
# ================================
class WaveModel:
    """
    A class representing the NFT wave equation for simulating neural activity using eigenmodes.

    Parameters
    ----------
    evecs : array_like
        Eigenvectors representing the spatial maps of the modes.
    evals : array_like
        Eigenvalues representing the frequencies of the modes.

    Attributes
    ----------
    r_s : float
        Length scale [mm]. Default value is 30.
    gamma_s : float
        Damping rate [ms^-1]. Default value is 116 * 1e-3.
    tstep : float
        Time step [ms]. Default value is 0.1.
    tmax : float
        Maximum time [ms]. Default value is 100.
    tspan : list
        Time period limits.
    T : array_like
        Time vector.

    Methods
    -------
    wave_ode(y, t, mode_coeff, eval)
        ODE function for solving the NFT wave equation.
    wave_fourier(mode_coeff, eval, T)
        Solve the NFT wave equation using a Fourier transform.
    solve(ext_input, method='Fourier')
        Simulate neural activity using eigenmodes.

    Notes
    -----
    This class and associated functions were adapted from the OHBM 2023 course on Whole-brain models:
    https://griffithslab.github.io/OHBM-whole-brain-modelling-course/materials/
    """

    def __init__(self, evecs, evals, cmean):
        # Default parameters
        # self.r_s = 28.9                             # in mm
        self.gamma_s = 116 * 1e-3                   # in ms^-1
        self.cmean = cmean                            # in mm/ms
        self.tstep = 0.1                            # in ms
        self.tmax = 100                             # in ms
        self.tspan = [0, self.tmax]
        self.T = np.arange(0, self.tmax + self.tstep, self.tstep)
        self.evecs = evecs
        self.evals = evals
        
    def solve(self, ext_input, solver_method='Fourier', eig_method='matrix', B=None):
        """
        Simulate neural activity using eigenmodes.

        Parameters
        ----------
        ext_input : array_like
            External input to wave model.
        solver_method : str, optional
            The method used for simulation. Can be either 'Fourier' (default) or 'ODE'.
        eig_method : str, optional
            The method used for the eigendecomposition. Default is 'matrix'.
        B : array-like, optional
            The mass matrix used for the eigendecomposition when method is 'orthonormal'.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays:
            - mode_activity: The simulated activity for each mode.
            - sim_activity: The combined simulated activity for all modes.
        """

        n_modes = np.shape(self.evecs)[1]

        # TODO: perform mode decomposition of external input outside of if statement
        if solver_method == 'ODE':
            ext_input_coeffs = calc_eigendecomposition(ext_input, self.evecs, eig_method, B=B)
            sim_activity = np.zeros((n_modes, np.shape(ext_input_coeffs)[1]))
            for mode_ind in range(n_modes):
                mode_coeff = ext_input_coeffs[mode_ind, :]
                eval = self.evals[mode_ind]
                yout = odeint(self.wave_ode, [mode_coeff[0], 0], self.T, args=(mode_coeff, eval))
                sim_activity[mode_ind, :] = yout[:, 0]

        elif solver_method == 'Fourier':
            # Append time vector with negative values to have a zero center
            T_append = np.concatenate((-np.flip(self.T[1:]), self.T))
            Nt = len(T_append)
            # Find the 0 index in the appended time vector
            t0_ind = np.argmin(np.abs(T_append))
            ext_input_coeffs_temp = calc_eigendecomposition(ext_input, self.evecs, eig_method, B=B)

            ext_input_coeffs = np.zeros((n_modes, Nt))
            ext_input_coeffs[:, t0_ind:] = ext_input_coeffs_temp

            sim_activity = np.zeros((n_modes, Nt))
            for mode_ind in range(n_modes):
                mode_coeff = ext_input_coeffs[mode_ind, :]
                eval = self.evals[mode_ind]
                yout = self.wave_fourier(mode_coeff, eval, T_append)
                sim_activity[mode_ind, :] = yout

            sim_activity = sim_activity[:, t0_ind:]

        mode_activity = sim_activity
        sim_activity = self.evecs @ sim_activity

        return mode_activity, sim_activity

    def wave_ode(self, y, t, mode_coeff, eval):
        """
        ODE function for solving the NFT wave equation.

        Parameters
        ----------
        y : array_like
            Array of shape (2, 1) representing the current activity and its first order derivative.
        t : float
            Current time.
        mode_coeff : array_like
            Coefficient of the mode at each time point.
        eval : float
            Eigenvalue of the mode.

        Returns
        -------
        array_like
            Array of shape (2, 1) representing the output activity and its first order derivative.
        """

        out = np.zeros(2)
        coef_interp = np.interp(t, self.T, mode_coeff)

        out[0] = y[1]
        out[1] = self.gamma_s**2 * (coef_interp - (2 / self.gamma_s) * y[1] - y[0] * (1 + self.r_s**2 * eval))

        return out
    
    def wave_fourier(self, mode_coeff, eval, T):
        """
        Solve the NFT wave equation using a Fourier transform.

        Parameters
        ----------
        mode_coeff : array_like
            Coefficient of the mode at each time point.
        eval : float
            Eigenvalue of the mode.
        T : array_like
            Time vector with zero at center.

        Returns
        -------
        array_like
            Solution of the wave equation.
        """

        Nt = len(T)
        Nw = Nt
        wsamp = 1 / np.mean(self.tstep) * 2 * np.pi
        jvec = np.arange(Nw)
        w = wsamp * 1 / Nw * (jvec - Nw / 2)

        # Calculate the -1 vectors needed for the Fourier transform
        wM = (-1) ** np.arange(1, len(w) + 1)

        # Perform the Fourier transform
        mode_coeff_fft = wM * np.fft.fft(wM * mode_coeff)

        # Solve the wave equation in Fourier space (this equation still works for the heterogeneous
        # helmholtz derviation since evals_new = cmean * evals_old)
        out_fft = self.gamma_s**2 * mode_coeff_fft / (-w**2 - 2*1j*w*self.gamma_s 
                                                      + self.gamma_s**2 + self.cmean**2*eval)

        # Perform the inverse Fourier transform
        out = np.real(wM * np.fft.ifft(wM * out_fft))

        return out


# ===================================
# ========== BALLOON MODEL ==========
# ===================================
class BalloonModel:
    def __init__(self, evecs):
        # Default independent model parameters (taken from Demirtas et al., 2019)
        self.V0 = 0.02      # resting blood volume fraction [unitless]
        self.kappa = 0.65   # signal decay rate [s^-1]
        self.gamma = 0.41   # rate of flow-dependent elimination [s^-1]
        self.tau = 0.98     # hemodynamic transit time [s]
        self.alpha = 0.32   # Grubb's exponent [unitless]
        self.rho = 0.34     # resting oxygen extraction fraction [unitless]
        self.k1 = 3.72
        self.k2 = 0.527
        self.k3 = 0.53      # Pang et al., 2023 used k3 = 0.48
        
        # Other parameters
        self.w_f = 0.56
        self.Q0 = 1
        self.rho_f = 1000
        self.eta = 0.3
        self.Xi_0 = 1
        self.beta = 3
        self.V_0 = 0.02
        self.beta = (self.rho + (1 - self.rho) * np.log(1 - self.rho)) / self.rho
        
        # Computational parameters
        self.tstep = 0.1  # time step
        self.tmax = 100  # maximum time
        
        # Dependent parameters
        self.tspan = [0, self.tmax]  # time period limits
        self.T = np.arange(0, self.tmax + self.tstep, self.tstep)  # time vector

        # Input parameters
        self.evecs = evecs

    def solve(self, neural, solver_method='ODE', eig_method='matrix', B=None):
        n_modes = self.evecs.shape[1]

        if solver_method == 'ODE':
            ext_input_coeffs = calc_eigendecomposition(neural, self.evecs, eig_method, B=B)
            F0 = np.tile(0.001*np.ones(n_modes), (4, 1)).T
            F = F0.copy()
            sol = {'z': np.zeros((n_modes, len(self.T))),
                'f': np.zeros((n_modes, len(self.T))),
                'v': np.zeros((n_modes, len(self.T))),
                'q': np.zeros((n_modes, len(self.T))),
                'BOLD': np.zeros((n_modes, len(self.T)))}
            sol['z'][:, 0] = F[:, 0]
            sol['f'][:, 0] = F[:, 1]
            sol['v'][:, 0] = F[:, 2]
            sol['q'][:, 0] = F[:, 3]

            for k in range(1, len(self.T)):
                dF = self.balloon_ode(F, self.T[k-1], ext_input_coeffs[:, k-1])
                F = F + dF*self.tstep
                sol['z'][:, k] = F[:, 0]
                sol['f'][:, k] = F[:, 1]
                sol['v'][:, k] = F[:, 2]
                sol['q'][:, k] = F[:, 3]

            sol['BOLD'] = 100*self.V0*(self.k1*(1 - sol['q']) 
                                       + self.k2*(1 - sol['q']/sol['v']) + self.k3*(1 - sol['v']))
            sim_activity = sol['BOLD']

        elif solver_method == 'Fourier':
            # Append time vector with negative values to have a zero center
            T_append = np.concatenate((-np.flip(self.T[1:]), self.T))
            Nt = len(T_append)
            # Find the 0 index in the appended time vector
            t0_ind = np.argmin(np.abs(T_append))

            # Mode decomposition of external input
            ext_input_coeffs_temp = calc_eigendecomposition(neural, self.evecs, eig_method, B=B)

            # Append external input coefficients for negative time values
            ext_input_coeffs = np.zeros((n_modes, Nt))
            ext_input_coeffs[:, t0_ind:] = ext_input_coeffs_temp

            sim_activity = np.zeros((n_modes, Nt))
            for mode_ind in range(n_modes):
                mode_coeff = ext_input_coeffs[mode_ind, :]
                yout = self.balloon_fourier(mode_coeff, T_append)
                sim_activity[mode_ind, :] = yout

            sim_activity = sim_activity[:, t0_ind:]

        mode_activity = sim_activity
        sim_activity = self.evecs @ sim_activity

        return mode_activity, sim_activity
    
    def balloon_ode(self, F, t, S):
        z = F[0]
        f = F[1]
        v = F[2]
        q = F[3]
        dF = np.zeros(4)
        dF[0] = S - self.kappa*z - self.gamma*(f - 1)
        dF[1] = z       
        dF[2] = (1/self.tau)*(f - v**(1/self.alpha))
        dF[3] = (1/self.tau)*((f/self.rho)*(1 - (1 - self.rho)**(1/f)) - q*v**(1/self.alpha - 1))

        return dF

    def balloon_fourier(self, mode_coeff, T):
        Nt = len(T)
        Nw = Nt
        wsamp = 1 / np.mean(self.tstep) * 2*np.pi
        jvec = np.arange(Nw)
        w = (wsamp) * 1/Nw * (jvec - Nw/2)

        # Calculate the -1 vectors needed for the Fourier transform
        wM = (-1) ** np.arange(1, len(w) + 1)

        # Perform the Fourier transform
        mode_coeff_fft = wM * np.fft.fft(wM * mode_coeff) 

        T_Fz = 1 / (-(w + 1j * 0.5 * self.kappa)**2 + self.w_f**2)
        T_yF = self.V_0 * (self.alpha * (self.k2 + self.k3) * (1 - 1j * self.tau * w) 
                            - (self.k1 + self.k2 ) *(self.alpha + self.beta - 1 
                            - 1j * self.tau * self.alpha * self.beta * w))/((1 - 1j * self.tau * w)
                            *(1 - 1j * self.tau * self.alpha * w))
        T_yz = T_yF * T_Fz
        out_fft = T_yz * mode_coeff_fft
        
        # Perform the inverse Fourier transform
        out = np.real(wM * np.fft.ifft(wM * out_fft))

        return out
    