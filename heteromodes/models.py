import numpy as np
import scipy as sp
from scipy.integrate import odeint
from heteromodes.eigentools import calc_eigendecomposition


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

    def __init__(self, evecs, evals, r=28.9, gamma=0.116, tstep=0.1, tmax=100):
        self.evecs = evecs
        self.evals = evals
        self.r = r                          # in mm
        self.gamma = gamma                 # in ms^-1
        self.tstep = tstep                           # in ms
        self.tmax = tmax                             # in ms
        self.t = np.arange(0, self.tmax + self.tstep, self.tstep)
        
    def solve(self, ext_input, solver_method='Fourier', eig_method='matrix', mass=None):
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
        mass : array-like, optional
            The mass matrix used for the eigendecomposition when method is 'orthogonal'.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays:
            - mode_activity: The simulated activity for each mode.
            - sim_activity: The combined simulated activity for all modes.
        """

        n_modes = np.shape(self.evecs)[1]

        # Calculate mode decomposition of external input
        ext_input_coeffs = calc_eigendecomposition(ext_input, self.evecs, eig_method, mass=mass)
        if solver_method == 'ODE':
            sim_activity = np.zeros((n_modes, len(self.t)))
            for mode_ind in range(n_modes):
                mode_coeff = ext_input_coeffs[mode_ind, :]
                yout = odeint(self.wave_ode, [mode_coeff[0], 0], self.t, args=(mode_coeff, self.evals[mode_ind]))
                sim_activity[mode_ind, :] = yout[:, 0]

        elif solver_method == 'Fourier':
            sim_activity = np.zeros((n_modes, len(self.t)))
            for mode_ind in range(n_modes):
                yout = self.wave_fourier(ext_input_coeffs[mode_ind, :], self.evals[mode_ind], self.t)
                sim_activity[mode_ind, :] = yout

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
        coef_interp = np.interp(t, self.t, mode_coeff)

        out[0] = y[1]
        out[1] = self.gamma**2 * (coef_interp - (2 / self.gamma) * y[1] - y[0] * (1 + self.r**2 * eval))

        return out
    
    def wave_fourier(self, mode_coeff, eval, t):
        """
        Solve the NFT wave equation using a Fourier transform.

        Parameters
        ----------
        mode_coeff : array_like
            Coefficient of the mode at each time point.
        eval : float
            Eigenvalue of the mode.
        t : array_like
            Time vector with zero at center.

        Returns
        -------
        array_like
            Solution of the wave equation.
        """

        q_hat = sp.fft.fft(mode_coeff)   # scipy fft is faster than numpy fft
        omega = sp.fft.fftfreq(len(t), t[1] - t[0]) * 2 * np.pi
        phi_hat = (self.gamma**2 * q_hat) / (-omega**2 + 2j*self.gamma*omega + self.gamma**2 + (self.gamma*self.r)**2 * eval)
        phi = np.real(sp.fft.ifft(phi_hat))

        return phi


class BalloonModel:
    def __init__(self, evecs, tstep=0.1, tmax=100):
        # Default independent model parameters
        self.kappa = 0.65   # signal decay rate [s^-1]
        self.gamma = 0.41   # rate of elimination [s^-1]
        self.tau = 0.98     # hemodynamic transit time [s]
        self.alpha = 0.32   # Grubb's exponent [unitless]
        self.rho = 0.34     # resting oxygen extraction fraction [unitless]
        self.V0 = 0.02      # resting blood volume fraction [unitless]
        
        # Other parameters
        self.w_f = 0.56
        self.Q0 = 1
        self.rho_f = 1000
        self.eta = 0.3
        self.Xi_0 = 1
        self.beta = 3
        self.V_0 = 0.02
        self.k1 = 3.72
        self.k2 = 0.527
        self.k3 = 0.48
        self.beta = (self.rho + (1 - self.rho) * np.log(1 - self.rho)) / self.rho
         
        # Computational parameters
        self.tstep = tstep  # time step
        self.tmax = tmax  # maximum time
        
        # Dependent parameters
        self.t = np.arange(0, self.tmax + self.tstep, self.tstep)  # time vector

        # Input parameters
        self.evecs = evecs

    def solve(self, neural, solver_method='ODE', eig_method='matrix', mass=None):
        n_modes = self.evecs.shape[1]

        if solver_method == 'ODE':
            ext_input_coeffs = calc_eigendecomposition(neural, self.evecs, eig_method, mass=mass)
            F0 = np.tile(0.001*np.ones(n_modes), (4, 1)).T
            F = F0.copy()
            sol = {'z': np.zeros((n_modes, len(self.t))),
                'f': np.zeros((n_modes, len(self.t))),
                'v': np.zeros((n_modes, len(self.t))),
                'q': np.zeros((n_modes, len(self.t))),
                'BOLD': np.zeros((n_modes, len(self.t)))}
            sol['z'][:, 0] = F[:, 0]
            sol['f'][:, 0] = F[:, 1]
            sol['v'][:, 0] = F[:, 2]
            sol['q'][:, 0] = F[:, 3]

            for k in range(1, len(self.t)):
                dF = self.balloon_ode(F, self.t[k-1], ext_input_coeffs[:, k-1])
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
            T_append = np.concatenate((-np.flip(self.t[1:]), self.t))
            Nt = len(T_append)
            # Find the 0 index in the appended time vector
            t0_ind = np.argmin(np.abs(T_append))

            # Mode decomposition of external input
            ext_input_coeffs_temp = calc_eigendecomposition(neural, self.evecs, eig_method, mass=mass)

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

    # TODO: simplify this function as in wave_fourier
    def balloon_fourier(self, mode_coeff, t):
        Nw = len(t)
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
    