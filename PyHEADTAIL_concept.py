#!/usr/bin/env python
# coding: utf-8

# # PyHEADTAIL test ground
# 
# Created Feb 2019, Adrian Oeftiger
# 
# ## Goal
# 
# The present notebook implements the basic functionality of the multi-particle beam dynamics simulation library `PyHEADTAIL`: https://github.com/PyCOMPLETE/PyHEADTAIL/ . The goal is to provide an environment for exploring concepts, how `PyHEADTAIL` can efficiently be accelerated. 
# 
# `PyHEADTAIL` is used to advance macro-particles through an accelerator, alternating between the single-particle transport and the multi-particle interaction nodes. Here we will only implement a specific interaction node based on wakefield interaction. These collective effects are based on (i.) a discretisation of the continuously distributed macro-particles onto a regular grid, (ii.) finding the solution of electromagnetic interaction on the grid typically via Green's functions using convolution (can exploit FFT algorithm) and (iii.) interpolation back to the particles.
# 
# With the implemented context management of `PyHEADTAIL`, the physics can be implemented in a single code passage, while targeting CPU (`numpy`/`cython`) and GPU (`PyCUDA`) hardware. This notebook explains and reflects this structure.
# 
# The following code snippets are partly extracted from the original PyHEADTAIL source authored to significant portions by Kevin Li, Adrian Oeftiger, Michael Schenk and Stefan Hegglin.
# 
# ## Structure
# 
# In the first part I., the essential features of `PyHEADTAIL` are implemented: 
# 1. Accelerator element and beam description containing the data
# 2. Context management with the math functions
# 3. Single-particle transport (embarrassingly parallel tracking)
# 4. Multi-particle interaction (memory-interactive collective effects)
# 
# The second part II. shows how a typical simulation is set up by a user in a script, using the library classes.
# 
# The third part III. profiles the code and shows that typically the discretisation and statistics computation of the distribution takes most time.
# 
# The fourth part IV. deals with possible approaches to accelerate the algorithms. Here we demonstrate how to use the context management approach by speeding up some statistics computations based on outsourcing them to `cython`. The context management also provides an easy means to accelerate on the graphical processing unit via the `cupy` library (`PyCUDA` would work in an equivalent way).
# 
# Here, further approaches such as `numba` can be tested.

# ## I. PyHEADTAIL abstraction

# ### 1. Basic description of accelerator and beam

# A part of an accelerator or a physics effect class knows how to track the particle beam:

# In[1]:


from abc import ABCMeta, abstractmethod

class Element(object):
    '''Abstract transporting element as part of the
    accelerator layout, knows how to /track/ a beam.
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def track(self, beam):
        '''Perform tracking of beam through this Element.'''
        pass


# The beam is represented by a class, the arrays `x`, `xp`, `y`, `yp`, `z` and `dp` are the coordinate and momentum values of each individual macro-particle:

# In[2]:


from scipy.constants import e, m_p, c

class Particles(object):
    '''Description of a beam of macro-particles with its coordinates
    and conjugate momenta as well as basic properties.
    '''
    def __init__(
            self, x, xp, y, yp, z, dp, intensity, gamma, circumference,
            charge=e, mass=m_p, *args, **kwargs):
        '''Args: x, y and z are the horizontal, vertical and longitudinal
        coordinate arrays of the macro-particles in the along with
        xp, yp and dp their dimensionless conjugate momenta.
        intensity denotes the number of real particles in the beam.
        gamma is the Lorentz energy assumed to be the same for all
        particles.
        circumference is the synchrotron circumference.
        '''
        self.x = x
        self.xp = xp
        self.y = y
        self.yp = yp
        self.z = z
        self.dp = dp
        self.coords_n_momenta = ['x', 'xp', 'y', 'yp', 'z', 'dp']
        
        self.macroparticlenumber = len(x)
        assert self.macroparticlenumber == len(xp)
        assert self.macroparticlenumber == len(y) 
        assert self.macroparticlenumber == len(yp)
        assert self.macroparticlenumber == len(z)
        assert self.macroparticlenumber == len(dp)
        
        self.intensity = intensity
        
        self.charge = charge
        self.mass = mass

        self.circumference = circumference
        self.gamma = gamma
        
    @property
    def particlenumber_per_mp(self):
        return self.intensity / self.macroparticlenumber
    @property
    def charge_per_mp(self):
        return self.particlenumber_per_mp * self.charge

    # energy formulae as properties
    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self.gamma**-2)
        self._betagamma = np.sqrt(self.gamma**2 - 1)
        self._p0 = self.betagamma * self.mass * c

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, value):
        self.gamma = 1. / np.sqrt(1 - value ** 2)

    @property
    def betagamma(self):
        return self._betagamma
    @betagamma.setter
    def betagamma(self, value):
        self.gamma = np.sqrt(value**2 + 1)

    @property
    def p0(self):
        return self._p0
    @p0.setter
    def p0(self, value):
        self.gamma = np.sqrt(1 + (value / (self.mass * c))**2)
    
    # DISTRIBUTION STATISTICS
    # for collective effects as well as storing the time evolution of the beam!
    # Note: will make use of the pmath library already (for context management),
    # see next section in this notebook
    
    ## centroids:
    def mean_x(self):
        return pm.mean(self.x)
    def mean_y(self):
        return pm.mean(self.y)
    def mean_z(self):
        return pm.mean(self.z)
    def mean_xp(self):
        return pm.mean(self.xp)
    def mean_yp(self):
        return pm.mean(self.yp)
    def mean_dp(self):
        return pm.mean(self.dp)
    
    ## beam sizes:
    def sigma_x(self):
        return pm.std(self.x)
    def sigma_y(self):
        return pm.std(self.y)
    def sigma_z(self):
        return pm.std(self.z)
    def sigma_xp(self):
        return pm.std(self.xp)
    def sigma_y(self):
        return pm.std(self.yp)
    def sigma_dp(self):
        return pm.std(self.dp)
    
    ## emittances:
    def epsn_x(self):
        return pm.emittance_geo(self.x, self.xp) * self.betagamma
    def epsn_y(self):
        return pm.emittance_geo(self.y, self.yp) * self.betagamma
    def epsn_z(self):
        return 4*np.pi * pm.emittance_geo(self.z, self.dp) * self.p0 / e


# ### 2. Context Management

# #### a. Context manager
# 
# Here the CPU context with the numpy library acting on CPU RAM allocated arrays is defined:

# In[3]:


import numpy as np
from functools import partial

# references to all relevant methods on CPU via numpy
cpu_dict = dict(
    sin=np.sin,
    cos=np.cos,
    tan=np.tan,
    exp=np.exp,
    sinh=np.sinh,
    cosh=np.cosh,
    tanh=np.tanh,
    sqrt=np.sqrt,
    abs=np.abs,
    floor=np.floor,
    real=np.real,
    imag=np.imag,
    fft=np.fft.fft,
    ifft=np.fft.ifft,
    convolve=partial(np.convolve, mode='valid'), # could also be implemented as ifft(fft * fft) e.g. for CuPy
    histogram=np.histogram,
    mean=np.mean,
    cov=lambda a, b: np.cov(a, b)[0, 1],
    std=lambda u: np.sqrt(cpu_dict['cov'](u, u)),
    emittance_geo=lambda u, up: np.sqrt(np.linalg.det(np.cov(u, up))),
    asarray=np.array,
    empty=np.empty,
    zeros=np.zeros,
    linspace=np.linspace,
    arange=np.arange,
    take=np.take,
    where=np.where,
    concatenate=np.concatenate,
    clip=np.clip,
)

class CPU(object):
    '''CPU context manager working with the numpy library.
    Here just to show the principle.
    '''
    
    def __init__(self, beam):
        self.beam = beam
        self.to_move = beam.coords_n_momenta

    def __enter__(self):
        # moving data to device
        for attr in self.to_move:
            coord = getattr(self.beam, attr)
            transferred = np.asarray(coord)
            setattr(self.beam, attr, transferred)

        # replace functions in general.math.py
        pm.update_active_dict(cpu_dict)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # here should be moving data back to host
        pm.update_active_dict(pm._default_function_dict)


# #### b. PyHEADTAIL math library which links to the contextual math library
# 
# Any other dictionary implementing the same functions as `cpu_dict` could be given to `pmath`:

# In[4]:


class pmath(object):
    '''Math library which links to math functions depending on
    currently spilled function dictionary (depending on context).
    '''
    def __init__(self, default_function_dict=cpu_dict):
        self._default_function_dict = default_function_dict
        self.update_active_dict(default_function_dict)
        
    def update_active_dict(self, function_dict):
        for func in function_dict:
            setattr(self, func, function_dict[func])

# global state here, in real PyHEADTAIL this would be a module
pm = pmath()


# ### 3. Trackers advancing the particles (embarrassingly parallel)
# 
# The trackers make use of the `pmath` calls to the contextual math library.

# In[5]:


class RFSystem(Element):
    '''Radio-frequency system kicking in the longitudinal
    direction.
    '''
    def __init__(self, harmonic, voltage, dphi_offset):
        '''Args: harmonic number and the RF voltage are required.
        dphi_offset is the phase offset of the RF wave.
        '''
        self.harmonic = harmonic
        self.voltage = voltage
        self.dphi_offset = dphi_offset

    def track(self, beam):
        amplitude = np.abs(beam.charge) * self.voltage / (beam.beta * c)
        phi = self.harmonic * (2 * np.pi * beam.z / beam.circumference - 
                               self.dphi_offset)

        delta_p = beam.dp * beam.p0
        delta_p += amplitude * pm.sin(phi)
        # beam.p0 += self.p_increment
        beam.dp = delta_p / beam.p0


# In[6]:


class TransportSegment(Element):
    '''Effective segment of the accelerator transporting the beam over 
    a fraction of the synchrotron, applying linear transverse betatron
    rotation and a longitudinal drift.
    '''
    def __init__(
            self, pathlength, alpha_c, dmu0_x, dmu0_y,
            # the following ones could also be ignored for the concept:
            alpha_x_s0, beta_x_s0, alpha_x_s1, beta_x_s1,
            alpha_y_s0, beta_y_s0, alpha_y_s1, beta_y_s1,
    ):
        '''Args: pathlength is the amount of the synchrotron circumference
        which this segment covers. 
        alpha_c denotes the linear momentum compaction factor.
        dmu0_x and dmu0_y denote the bare betatron phase advance
        (in units of 2pi dQ) across this segment in the horizontal
        and vertical plane, respectively.
        alpha_x/y and beta_x/y denote the corresponding optics Twiss
        functions for x=horizontal and y=vertical and the subscripts
        s0 and s1 stand for initial and final value along the segment.
        '''
        # longitudinal:
        self.pathlength = pathlength
        self.alpha_c = alpha_c
        # horizontal:
        self.alpha_x_s0 = alpha_x_s0
        self.alpha_x_s1 = alpha_x_s1
        self.beta_x_s0 = beta_x_s0
        self.beta_x_s1 = beta_x_s1
        # vertical:
        self.alpha_y_s0 = alpha_y_s0
        self.alpha_y_s1 = alpha_y_s1
        self.beta_y_s0 = beta_y_s0
        self.beta_y_s1 = beta_y_s1
        
        # in general transverse
        self.dmu0_x = dmu0_x
        self.dmu0_y = dmu0_y
        # Floquet matrix:
        self.I, self.J = _build_segment_map(
            alpha_x_s0, beta_x_s0, alpha_x_s1, beta_x_s1,
            alpha_y_s0, beta_y_s0, alpha_y_s1, beta_y_s1,
        )
        
    def eta(self, gamma):
        return self.alpha_c - gamma**-2

    def track(self, beam):
        # longitudinal drift:
        beam.z -= self.eta(beam.gamma) * beam.dp * self.pathlength
        # transverse rotation:
        (M00, M01, M10, M11,
         M22, M23, M32, M33) = self.assemble_betatron_matrix(beam)
        x = M00 * beam.x + M01 * beam.xp
        xp = M10 * beam.x + M11 * beam.xp
        y = M22 * beam.y + M23 * beam.yp
        yp = M32 * beam.y + M33 * beam.yp
        beam.x = x
        beam.y = y
        beam.xp = xp
        beam.yp = yp

    def assemble_betatron_matrix(self, beam):
        dmu_x = self.dmu0_x
        dmu_y = self.dmu0_y
#         dmu_x = self.dmu0_x + pm.zeros(beam.x.shape, beam.x.dtype)
#         dmu_y = self.dmu0_y + pm.zeros(beam.y.shape, beam.y.dtype)
#         dmu_x = pm.asarray(self.dmu0_x)
#         dmu_y = pm.asarray(self.dmu0_y)
        
        # detuning still missing! --> detuning stuff / 2*np.pi
        
        # --> for the moment (PyCUDA compatibility) run numpy hardcoded on the float dmu0
        pm = np
        
        s_x = pm.sin(dmu_x)
        c_x = pm.cos(dmu_x)
        s_y = pm.sin(dmu_y)
        c_y = pm.cos(dmu_y)
        
        M00 = self.I[0, 0] * c_x + self.J[0, 0] * s_x
        M01 = self.I[0, 1] * c_x + self.J[0, 1] * s_x
        M10 = self.I[1, 0] * c_x + self.J[1, 0] * s_x
        M11 = self.I[1, 1] * c_x + self.J[1, 1] * s_x
        M22 = self.I[2, 2] * c_y + self.J[2, 2] * s_y
        M23 = self.I[2, 3] * c_y + self.J[2, 3] * s_y
        M32 = self.I[3, 2] * c_y + self.J[3, 2] * s_y
        M33 = self.I[3, 3] * c_y + self.J[3, 3] * s_y
        return M00, M01, M10, M11, M22, M23, M32, M33

def _build_segment_map(alpha_x_s0, beta_x_s0, alpha_x_s1, beta_x_s1,
                       alpha_y_s0, beta_y_s0, alpha_y_s1, beta_y_s1):
    '''Calculate Floquet transformation matrices I and J which only
    depend on the TWISS parameters at the boundaries of the
    accelerator segment.
    alpha_x/y and beta_x/y denote the corresponding optics Twiss
    functions for x=horizontal and y=vertical and the subscripts
    s0 and s1 stand for initial and final value along the segment.
    '''
    I = np.zeros((4, 4))
    J = np.zeros((4, 4))

    # Sine component.
    I[0, 0] = np.sqrt(beta_x_s1 / beta_x_s0)
    I[0, 1] = 0.
    I[1, 0] = (np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
               (alpha_x_s0 - alpha_x_s1))
    I[1, 1] = np.sqrt(beta_x_s0 / beta_x_s1)
    I[2, 2] = np.sqrt(beta_y_s1 / beta_y_s0)
    I[2, 3] = 0.
    I[3, 2] = (np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
               (alpha_y_s0 - alpha_y_s1))
    I[3, 3] = np.sqrt(beta_y_s0 / beta_y_s1)

    # Cosine component.
    J[0, 0] = np.sqrt(beta_x_s1 / beta_x_s0) * alpha_x_s0
    J[0, 1] = np.sqrt(beta_x_s0 * beta_x_s1)
    J[1, 0] = -(np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
                (1. + alpha_x_s0 * alpha_x_s1))
    J[1, 1] = -np.sqrt(beta_x_s0 / beta_x_s1) * alpha_x_s1
    J[2, 2] = np.sqrt(beta_y_s1 / beta_y_s0) * alpha_y_s0
    J[2, 3] = np.sqrt(beta_y_s0 * beta_y_s1)
    J[3, 2] = -(np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
                (1. + alpha_y_s0 * alpha_y_s1))
    J[3, 3] = -np.sqrt(beta_y_s0 / beta_y_s1) * alpha_y_s1
    
    return I, J


# ### 4. Collective effects

# Electromagnetic interaction between the macro-particles is often implemented by a discretisation of the beam distribution into coarsely distributed grid points. Then the physics are solved on this regular grid (typically with a Green's function type approach) and interpolated back to yield the kicks of the macro-particles.

# #### Wakefield

# ##### (i.) Slicing

# Wakefields require a longitudinal discretisation of the beam into slices. Each of these slices needs a mean value per transverse plane `x` and `y` of the contained particles:

# In[7]:


class SliceSet(object):
    '''Container of slices with the histogram and statistics
    for each slice.
    '''
    def __init__(self, z_bins, n_macroparticles_per_slice, 
                 slice_index_of_particle, mean_x, mean_y):
        self.n_slices = len(n_macroparticles_per_slice)
        self.z_bins = z_bins
        self.n_macroparticles_per_slice = n_macroparticles_per_slice
        self.slice_index_of_particle = slice_index_of_particle
        self.mean_x = mean_x
        self.mean_y = mean_y

    def convert_to_particles(self, slice_array):
        '''Distribute slice_array entries with values per slice
        to particles based on which particle sits in which slice.
        All particles outside of z_bins are assigned 0.
        '''
        ids = self.slice_index_of_particle
        particle_array = pm.zeros(ids.shape, dtype=np.float64)
        p_id = pm.where((0 <= ids) & (ids < self.n_slices))[0]
        s_id = pm.take(ids, p_id)
        particle_array[p_id] = pm.take(slice_array, s_id)
        return particle_array

class Slicer(object):
    '''Longitudinally slices up the beam into uniform bins (slices)
    and computes distribution statistics for each slice.
    '''
    def __init__(self, n_slices, z_min, z_max):
        '''Args: The slicing interval along z is defined by (z_min, z_max),
        along which n_slices longitudinal bins are distributed.
        '''
        self.n_slices = n_slices
        self.z_min, self.z_max = z_min, z_max
        
    def slice(self, beam):
        '''Factory method for SliceSets, computing discretisation and
        statistics.
        '''
        z_bins = pm.linspace(self.z_min, self.z_max, self.n_slices + 1)
        hist, _ = pm.histogram(beam.z, bins=z_bins) # _ == z_bins
        ids = self.particles_to_slices(beam)
        mean_x, mean_y = self.compute_means(ids, beam.x, beam.y)
        return SliceSet(
            z_bins=z_bins,
            n_macroparticles_per_slice=hist,
            slice_index_of_particle=ids,
            mean_x=mean_x,
            mean_y=mean_y,
        )

    def particles_to_slices(self, beam):
        '''Compute and return slice id for each particle.'''
#         (beam.z - self.z_min) * self.n_slices // (self.z_max - self.z_min)
        ids = pm.floor(((beam.z - self.z_min) * self.n_slices) / (self.z_max - self.z_min))
        return ids.astype(np.int32)

    def compute_means(self, slice_index_of_particle, x, y):
        mean_x = pm.zeros(self.n_slices, dtype=np.float64)
        mean_y = pm.zeros(self.n_slices, dtype=np.float64)
        for i in range(self.n_slices):
            p_id = pm.where(slice_index_of_particle == i)[0]
            if any(p_id):
                mean_x[i] = pm.mean(x[p_id])
                mean_y[i] = pm.mean(y[p_id])
        return mean_x, mean_y


# ##### (ii.) Broadband resonator

# In[8]:


class BroadBandResonator(object):
    '''Applies the transverse dipolar wakefield of a circular
    broad-band resonator to the beam.
    '''
    def __init__(self, slicer, R_shunt, frequency, Q):
        '''Args: shunt impedance R_shunt, frequency and quality factor Q
        of a circular broad-band resonator are required.
        The slicer contains the longitudinal discretisation parameters.
        '''
        self.slicer = slicer
        self.Q = Q
        
        omega = 2 * np.pi * frequency
        
        self.prefactor = R_shunt * omega**2 / Q
        self.alpha = omega / (2 * Q)
        self.omegabar = pm.sqrt(pm.abs(omega**2 - self.alpha**2))

    def track(self, beam):
        slices = self.slicer.slice(beam)
        
        kick_factor = self.kick_factor(beam)
        
        # Green's function
        wake = self.extract_wake(slices)
        
        # slice quantities
        moment_x = slices.n_macroparticles_per_slice * slices.mean_x
        moment_y = slices.n_macroparticles_per_slice * slices.mean_y
        kicks_x = pm.convolve(moment_x, wake) * kick_factor
        kicks_y = pm.convolve(moment_y, wake) * kick_factor
        
        beam.xp += slices.convert_to_particles(kicks_x)
        beam.yp += slices.convert_to_particles(kicks_y)
    
    def wake_function(self, dt_to_target_slice):
        '''Resonator formula from A. Chao (Eq. 2.82).'''
        dt = pm.clip(dt_to_target_slice, a_min=None, a_max=0)
        y = self.prefactor * pm.exp(self.alpha * dt)
        if self.Q > 0.5:
            y *= pm.sin(self.omegabar * dt) / self.omegabar
        elif self.Q == 0.5:
            y *= dt
        else:
            y *= pm.sinh(self.omegabar * dt) / self.omegabar
        return y
    
    def extract_wake(self, slices):
        s = slices
        z_centers = s.z_bins[:-1] + 0.5 * (s.z_bins[1:] - s.z_bins[:-1])
        dt = z_centers / (beam.beta * c)
        
        dt_to_target_slice = pm.concatenate(
            (dt - dt[-1], (dt - dt[0])[1:]))
        
        return self.wake_function(dt_to_target_slice)
    
    @staticmethod
    def kick_factor(beam):
        return (-(beam.charge)**2 / (beam.p0 * beam.beta * c) * 
                beam.particlenumber_per_mp)


# ## II. Let's set up a "user" simulation

# In[9]:



# Creating the beam instance as a Gaussian distribution with $1\,000\,000$ macro-particles:

# In[10]:


def get_beam(n_mp=int(1e6), seed=1500000000):
    np.random.seed(seed)
    x = np.random.normal(scale=1e-3, size=n_mp)
    xp = np.random.normal(scale=1e-3, size=n_mp)
    y = np.random.normal(scale=1e-3, size=n_mp)
    yp = np.random.normal(scale=1e-3, size=n_mp)
    z = np.random.normal(scale=0.05, size=n_mp)
    dp = np.random.normal(scale=1e-4, size=n_mp)

    beam = Particles(x, xp, y, yp, z, dp, 
                     intensity=4e11, gamma=26, 
                     circumference=1100*2*np.pi)
    return beam


# ### II. 1. Just the tracking without multi-particle interaction

# Creating the one turn map around the accelerator ring with some physics parameters:

# In[11]:


harmonic = 4620
voltage = 4.5e6
alpha_c = 18**-2

Q_x = 20.2
Q_y = 20.18


# In[12]:


half_betatron_map = TransportSegment(
    beam.circumference / 2, alpha_c, 
    2*np.pi * Q_x / 2, 2*np.pi * Q_y / 2, 
    0, 1, 0, 1, 0, 1, 0, 1)

rf_systems = RFSystem(harmonic, voltage, dphi_offset=0)


# In[13]:


one_turn_map = [half_betatron_map, rf_systems, half_betatron_map]


# Some sanity checks:

# In[14]:


rot_mat = half_betatron_map.assemble_betatron_matrix(beam)

assert 1 == np.linalg.det(
    np.matrix(rot_mat[:4]).reshape((2, 2)))

assert 1 == np.linalg.det(
    np.matrix(rot_mat[4:]).reshape((2, 2)))


# ### II. 2. Tracking with wakefields

# In[15]:


slicer = Slicer(n_slices=500, z_min=-1, z_max=1)

resonator = BroadBandResonator(slicer, R_shunt=7e6, frequency=1.3e9, Q=1)


# In[16]:


one_turn_map_with_wf = one_turn_map + [resonator]
