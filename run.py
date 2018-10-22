import numpy as np
import math
import itertools as itool
import matplotlib.pyplot as plot

epsilon = 1#0**-21
sigma = 1#0**-8


class particle:
    def __init__(self, dim, i):
        self.name = i
        self.position = np.zeros(dim,  dtype='longdouble')
        self.velocity = np.zeros(dim,  dtype='longdouble')
        self.force = np.zeros(dim,  dtype='longdouble')
        self.masses = np.ones(dim,  dtype='longdouble')

class MolecSimulation_MC:
    '''Class for setting up simple Molecular Dynamics simulation'''
    def __init__(self, N, dim, particles):
        '''N is the number of particles'''
        self.NumParticles = N
        self.dim = dim
        self.temperature = 0
        self.particles = particles
        self.particle_table = list(itool.combinations(self.particles, 2))
        self.pbc = False
        self.pressure = 0
        #self.conf = np.zeros((N, dim), dtype='longdouble')
        #self.forces = np.zeros((N, dim), dtype='longdouble')
        #self.cons_temp = 0
        #self.energy = np.
    
    def set_pbc(self,Lx,Ly,Lz):
        '''Set PBC'''
        self.pbc = True
        self.box_L = np.array([Lx, Ly, Lz], dtype='longdouble')
        return 
    
    def set_temperature(self, temperature):
        ''' set temperature '''
        self.cons_temp = 1
        self.temperature = temperature
        #self.kT = k_B*temperature
        self.kT = 2
        
    def config_initialise(self,init_type):
        '''function for initialisation of position'''
        if init_type:
            for pt in range(0,self.NumParticles):
                #print('initalize pos for particle: ',pt)
                if self.pbc:
                    self.particles[pt].position = np.random.random_sample(3)*self.box_L 
                    #self.conf[pt] = np.random.random_sample(3)*self.box_L
                else:
                    set_boxL = 10.0
                    self.particles[pt].position = np.random.random_sample(3)*set_boxL
                    #self.conf[pt] = np.random.random_sample(3)*set_boxL
        else:
            print('Given option:',init_type,'is not coded yet!')
        
        #return 1
    def minimum_image_creater(self, p):
        #b = {'length':0,'width':0,'height':0}
        length = self.box_L[0]
        width = self.box_L[1]
        height = self.box_L[2]
        #
        # b = self.box_L 
        #a = 
        return ([p.position[0] + length, p.position[1], p.position[2]],                 [p.position[0] - length, p.position[1], p.position[2]],                 [p.position[0], p.position[1] + width, p.position[2]],                  [p.position[0], p.position[1] - width, p.position[2]],                  [p.position[0], p.position[1], p.position[2] + height],                 [p.position[0], p.position[1], p.position[2] - height],                 [p.position[0], p.position[1], p.position[2]])

    def compute_pairwise_force(self, pt1, pt2):
        '''compute the eneergy of interaction between pt1 and pt2'''
        #r12 vector
        # apply periodic boundary condititons if necessary
        if self.pbc:
            r12_vec = min(list(map(lambda x: np.linalg.norm(pt1.position-x), self.minimum_image_creater(pt2))))
        else:
            r12_vec = np.linalg.norm(pt1.position - pt2.position)

        d_inv6 = (sigma/r12_vec)**6
        return (48*epsilon*d_inv6*(d_inv6-0.5))

    def compute_pairwise_energy(self, pt1, pt2):
        '''compute the eneergy of interaction between pt1 and pt2'''
        #r12 vector
        # apply periodic boundary condititons if necessary
        #print(type(self.minimum_image_creater(pt2)))
        #for i in a:
        #    print (list(i)) 
        '''
            images_array = list(self.minimum_image_creater(self.conf[pt2]))
            ind = 0
            sumer = np.linalg.norm(self.conf[pt1] - self.conf[pt2])
            for i in range(len(images_array)):
                if np.linalg.norm(self.conf[pt1] - images_array[i]) < sumer:
                    ind = i

            r12_vet = images_array[i]
        '''
        if self.pbc:
            r12_vec = min(list(map(lambda x: np.linalg.norm(pt1.position-x), self.minimum_image_creater(pt2))))
        else:
            r12_vec = np.linalg.norm(pt1.position - pt2.position)

        d_inv6 = (sigma/r12_vec)**6
        return  4*epsilon*d_inv6*(d_inv6-1)
    
    def compute_energy(self):
        '''use the positions and find the Potential Enenergy'''
        energy = 0
        for pair in self.particle_table:
            energy += 2*self.compute_pairwise_energy(pair[0],pair[1])
            #energy += self.compute_pairwise_energy(pair[1],pair[0])
        return energy
        # use compute pairwise_energy to compute total energy
    
    def compute_force(self):
        '''use the positions and find the Potential Enenergy'''
        force = 0
        for pair in self.particle_table:
            force += self.compute_pairwise_force(pair[0], pair[1])
        return force
        # use compute pairwise_energy to compute total energy

    def MC_Translate(self,max_displacement):
        '''given the max_displacement, move a randomly selected partcle by 
        random amount in all three direction'''
        # randomly select a particle
        #pt = long(np.random.random()*self.NumParticles)
        pt = np.random.randint(1,self.NumParticles)
        # random displacement
        disp = np.random.uniform(-1, 1, self.dim) * max_displacement
        e1 = self.compute_energy()
        
        self.particles[pt].position += disp
        temp = self.particles[pt].position
        self.particles[pt].position %= self.box_L
        
        e2 = self.compute_energy()
        #print(e1-e2)
        lada = 1
        p_accept = 1

        if e2-e1 > 0:
            lada = math.e**(-self.kT * (e2 - e1))
            p_accept = min(1,lada)
        
        #print(e2, e1, e2-e1, lada)   
        if p_accept != 1 and p_accept <= np.random.random():
            #print('trrr')
            self.particles[pt].position = temp
            self.particles[pt].position -= disp
        
        # find energy of interaction of particle "pt" in current config, pe_c
        # displace the particle
        #self.conf[pt] += disp
        # find enery of interaction of particle "pt" in new config, pe_n
        # metropolic monte carlo
        #dE = pe_n- pe_c
        #return accept
        
        vol = np.prod(self.box_L)
        force = self.compute_force() * 10**-10
        pressure = self.kT*self.NumParticles/vol + force/(vol*self.dim)
        #print(force)
        return pressure
    def MC_VolumeMove(self,max_fraction):
        '''given the maximum change in the volume/box-length allowed
           perform the MC move for volume change'''
        
        pe_c = self.compute_energy()
        #pressure
        # volume move
        frac = 1 + np.random.uniform(-1, 1, self.dim) * max_fraction
        #print(frac)
        vol1 = np.prod(self.box_L)
        self.box_L *= frac
        vol2 = np.prod(self.box_L)
        for pt in self.particles:
            pt.position *= frac
        #map(lambda x: x.position*=frac, self.particles)
        #self.conf *= frac
        pe_n = self.compute_energy()
        diff_e = pe_n - pe_c
        diff_v = vol2 - vol1
        #lada = -self.kT * ((diff_e + self.pressure * (diff_v)))
        #lada1 = - self.NumParticles * math.log(vol2/vol1)
        #print(lada, lada1)
        #force = self.compute_force() * 10**-10
        #self.pressure = (self.kT * N / vol2) + (force / (vol2 * self.dim))
        p_accept = 1
        lada = -self.kT * (diff_e + self.pressure * (diff_v)) - self.NumParticles * math.log(vol2 / vol1)

        if lada < 0:
            p_accept = min(1, math.e**(lada))

        if p_accept != 1 and p_accept <= np.random.random():
            self.box_L /= frac
            for pt in self.particles:
                pt.position /= frac
        mass = 1
        
        return self.NumParticles*mass/np.prod(self.box_L)

        #vol = np.prod(self.box_L)
        # metropolis monte carlo for volume move
        #return self.pressure

       
N = 100
dim = 3
p_array = [particle(dim, i) for i in range(N)]
s1 = MolecSimulation_MC(N, dim, p_array)
b = 10#**-8
s1.set_pbc(b,b,b)
s1.set_temperature(300)
s1.config_initialise('random')
ncycles = 10

pressures = np.array([], dtype='longdouble')
densities = np.array([], dtype='longdouble')
pressures_list = []
densities_list = []
#for i in s1.particle_table:
#    print(i[0].name,i[1].name)
print('NVT or NPT?')
a = input()

if a == 'nvt':
    for i in range(ncycles):
        a = s1.MC_Translate(b/100)
        if len(pressures) > 0 and len(pressures) % 10 == 0:
            pressures = np.delete(pressures,0)  #np.array([], dtype='longdouble')
        pressures = np.append(pressures, a)
        pressures_list.append(np.mean(pressures))
        print(i, np.mean(pressures))
    
    plot.plot(range(0, len(pressures_list), 1), pressures_list, label='Pressure')
    plot.legend()
    plot.show()

elif a == 'npt':
    s1.pressure = s1.MC_Translate(b / 100)
    for i in range(ncycles):
        if len(densities) > 0 and len(densities) % 10 == 0:
            densities = np.delete(densities,0) #np.array([], dtype='longdouble')
        if len(pressures) > 0 and len(pressures) % 10 == 0:
            pressures = np.delete(pressures,0)  #np.array([], dtype='longdouble')
            
        if np.random.uniform(0.0,0.03) < 1/(s1.NumParticles+1):
            a1 = s1.MC_VolumeMove(1)
            densities = np.append(densities, a1)
            densities_list.append(a1)
            print(i, np.mean(densities))
        else:
            a2 = s1.MC_Translate(b / 100)
            pressures = np.append(pressures, a2)
            pressures_list.append(np.mean(pressures))
    
    plot.plot(range(0, len(densities_list), 1), densities_list, label='Density')
    plot.legend()
    plot.show()
            #print(i, np.mean(pressures))
else:
    print('Error')


#for i in p_array:

#    print(i.name)
