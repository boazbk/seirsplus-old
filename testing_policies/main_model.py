from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as networkx
import numpy as numpy
import scipy as scipy
import scipy.integrate
import types
import random



#################################################################################################################################################
# Main stochastic model

class SEIRSNetworkModel():
    """
    A class to simulate the SEIRS Stochastic Network Model
    ===================================================
    Params: G       Network adjacency matrix (numpy array) or Networkx graph object.
            beta    Rate of transmission (exposure) (global)
            beta_local    Rate(s) of transmission (exposure) for adjacent individuals (optional)
            sigma   Rate of infection (upon exposure) 
            gamma   Rate of recovery (upon infection) 
            xi      Rate of re-susceptibility (upon recovery)  
            mu_I    Rate of infection-related death  
            mu_0    Rate of baseline death   
            nu      Rate of baseline birth
            p       Probability of interaction outside adjacent nodes
            
            Q       Quarantine adjacency matrix (numpy array) or Networkx graph object.
            beta_D  Rate of transmission (exposure) for individuals with detected infections (global)
            beta_local    Rate(s) of transmission (exposure) for adjacent individuals with detected infections (optional)
            sigma_D Rate of infection (upon exposure) for individuals with detected infections
            gamma_D Rate of recovery (upon infection) for individuals with detected infections
            mu_D    Rate of infection-related death for individuals with detected infections
            theta_E Rate of baseline testing for exposed individuals
            theta_I Rate of baseline testing for infectious individuals
            phi_E   Rate of contact tracing testing for exposed individuals
            phi_I   Rate of contact tracing testing for infectious individuals
            psi_E   Probability of positive test results for exposed individuals
            psi_I   Probability of positive test results for exposed individuals
            q       Probability of quarantined individuals interaction outside adjacent nodes
            
            initE   Init number of exposed individuals       
            initI   Init number of infectious individuals      
            initD_E Init number of detected infectious individuals
            initD_I Init number of detected infectious individuals   
            initR   Init number of recovered individuals     
            initF   Init number of infection-related fatalities
                    (all remaining nodes initialized susceptible)

            p_extern Probability of spontaneous infection
            p_periodic Prob of period test
            period   Period for resting

            batch  True - periodic testing done in batches of p_periodic*numNodes new people

            min_time : minimum time period to pass between testing same person twice
            
            count_non_random : count tests apart from random routine tests
            
            policy:  policy function
    """
    
    def policy(self):
        pass

    def __init__(self, G, beta, sigma, gamma, xi=0, mu_I=0, mu_0=0, nu=0, beta_local=None, p=0,
                    Q=None, beta_D=None, sigma_D=None, gamma_D=None, mu_D=None, beta_D_local=None,
                    theta_E=0, theta_I=0, phi_E=0, phi_I=0, psi_E=1, psi_I=1, q=0,
                    initE=0, initI=10, initD_E=0, initD_I=0, initR=0, initF=0,
                    node_groups=None, store_Xseries=False, p_extern = 0 , p_periodic =0, period=1, batch = True, min_time = 1 , count_non_random = False, policy = None, test_recovered = False, initT = 0 ):
        self.has_policy = False
        if policy:
            self.has_policy = True
            policy.__name__ = "policy"
            self.policy = types.MethodType(policy,self)
            
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Adjacency matrix:
        self.update_G(G)
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Quarantine Adjacency matrix:
        if(Q is None):
            Q = G # If no Q graph is provided, use G in its place
        self.update_Q(Q)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.parameters = { 'beta':beta, 'sigma':sigma, 'gamma':gamma, 'xi':xi, 'mu_I':mu_I, 'mu_0':mu_0, 'nu':nu, 
                            'beta_D':beta_D, 'sigma_D':sigma_D, 'gamma_D':gamma_D, 'mu_D':mu_D, 
                            'beta_local':beta_local, 'beta_D_local':beta_D_local, 'p':p,'q':q,
                            'theta_E':theta_E, 'theta_I':theta_I, 'phi_E':phi_E, 'phi_I':phi_I, 'psi_E':psi_E, 'psi_I':psi_I ,
                            'p_extern': p_extern, 'p_periodic':p_periodic , "period":period, "batch":batch, "min_time" : min_time, 
                            "count_non_random": count_non_random, "test_recovered":test_recovered }

        self.init_parameters = dict(self.parameters)
        self.update_parameters()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo up to 4 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*4 events/timesteps expected; initialize numNodes*5 timestep slots to start 
        # (will be expanded during run if needed)
        self.tseries    = numpy.zeros(5*self.numNodes)
        self.numE       = numpy.zeros(5*self.numNodes)
        self.numI       = numpy.zeros(5*self.numNodes)
        self.numD_E     = numpy.zeros(5*self.numNodes)
        self.numD_I     = numpy.zeros(5*self.numNodes)
        self.numR       = numpy.zeros(5*self.numNodes)
        self.numF       = numpy.zeros(5*self.numNodes)
        self.numS       = numpy.zeros(5*self.numNodes)
        self.N          = numpy.zeros(5*self.numNodes)
        self.numTested =  numpy.zeros(5*self.numNodes)
        self.numPositive = numpy.zeros(5 * self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if isinstance(initT, (int, float)):
            self.t          =  initT
        else:
            self.t = random.uniform(initT[0], initT[1])
        self.tmax       = 0 # will be set when run() is called
        self.tidx       = 0
        self.tseries[0] = self.t
        self.wait_until_t = 0
        self.currentR = 0 

        if (node_groups):
            for groupName in node_groups:
                self.nodeToTest[groupName] = 0
        else:
            self.nodeToTest = 0
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.numE[0]    = int(initE)
        self.numI[0]    = int(initI)
        self.numD_E[0]  = int(initD_E)
        self.numD_I[0]  = int(initD_I)
        self.numR[0]    = int(initR)
        self.numF[0]    = int(initF)
        self.numS[0]    = self.numNodes - self.numE[0] - self.numI[0] - self.numD_E[0] - self.numD_I[0] - self.numR[0] - self.numF[0]
        self.N[0]       = self.numS[0] + self.numE[0] + self.numI[0] + self.numD_E[0] + self.numD_I[0] + self.numR[0]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S          = 1
        self.E          = 2
        self.I          = 3
        self.D_E        = 4
        self.D_I        = 5
        self.R          = 6
        self.F          = 7

        self.X = numpy.array([self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) + [self.I]*int(self.numI[0]) + [self.D_E]*int(self.numD_E[0]) + [self.D_I]*int(self.numD_I[0]) + [self.R]*int(self.numR[0]) + [self.F]*int(self.numF[0])).reshape((self.numNodes,1))
        numpy.random.shuffle(self.X)

        self.store_Xseries = store_Xseries
        if(store_Xseries):
            self.Xseries        = numpy.zeros(shape=(5*self.numNodes, self.numNodes), dtype='uint8')
            self.Xseries[0,:]   = self.X.T

        self.transitions =  { 
                                'StoE': {'currentState':self.S, 'newState':self.E},
                                'EtoI': {'currentState':self.E, 'newState':self.I},
                                'ItoR': {'currentState':self.I, 'newState':self.R},
                                'ItoF': {'currentState':self.I, 'newState':self.F},
                                'RtoS': {'currentState':self.R, 'newState':self.S},
                                'EtoDE': {'currentState':self.E, 'newState':self.D_E},
                                'ItoDI': {'currentState':self.I, 'newState':self.D_I},
                                'DEtoDI': {'currentState':self.D_E, 'newState':self.D_I},
                                'DItoR': {'currentState':self.D_I, 'newState':self.R},
                                'DItoF': {'currentState':self.D_I, 'newState':self.F},
                                '_toS': {'currentState':True, 'newState':self.S},
                                'StoNS': {'currentState':self.S, 'newState': self.S }

                            }
        self.last_tested = numpy.zeros((self.numNodes, 1)) - 100 # everybody has a fake last tested time of -100 days 
        self.time_detected = 0
        self.small_step = False

        self.count_non_random = count_non_random
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node subgroup data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if(node_groups):
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   numpy.array(nodeList),
                                                 'mask':    numpy.isin(range(self.numNodes), nodeList).reshape((self.numNodes,1))}
                self.nodeGroupData[groupName]['numS']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numE']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numI']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD_E']     = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD_I']     = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numR']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numF']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['N']          = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numTested'] = numpy.zeros(5 * self.numNodes)
                self.nodeGroupData[groupName]['numS'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numD_E'][0]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_E)
                self.nodeGroupData[groupName]['numD_I'][0]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_I)
                self.nodeGroupData[groupName]['numR'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['N'][0]       = self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numD_E'][0] + self.nodeGroupData[groupName]['numD_I'][0] + self.nodeGroupData[groupName]['numR'][0]

         

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_parameters(self):
        import time
        updatestart = time.time()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta           = numpy.array(self.parameters['beta']).reshape((self.numNodes, 1))  if isinstance(self.parameters['beta'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta'], shape=(self.numNodes,1))
        self.sigma          = numpy.array(self.parameters['sigma']).reshape((self.numNodes, 1)) if isinstance(self.parameters['sigma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma'], shape=(self.numNodes,1))
        self.gamma          = numpy.array(self.parameters['gamma']).reshape((self.numNodes, 1)) if isinstance(self.parameters['gamma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma'], shape=(self.numNodes,1))
        self.xi             = numpy.array(self.parameters['xi']).reshape((self.numNodes, 1))    if isinstance(self.parameters['xi'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['xi'], shape=(self.numNodes,1))
        self.mu_I           = numpy.array(self.parameters['mu_I']).reshape((self.numNodes, 1))  if isinstance(self.parameters['mu_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_I'], shape=(self.numNodes,1))
        self.mu_0           = numpy.array(self.parameters['mu_0']).reshape((self.numNodes, 1))  if isinstance(self.parameters['mu_0'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_0'], shape=(self.numNodes,1))
        self.nu             = numpy.array(self.parameters['nu']).reshape((self.numNodes, 1))    if isinstance(self.parameters['nu'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['nu'], shape=(self.numNodes,1))
        self.p              = numpy.array(self.parameters['p']).reshape((self.numNodes, 1))     if isinstance(self.parameters['p'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['p'], shape=(self.numNodes,1))
        self.p_extern       =  numpy.array(self.parameters['p_extern']).reshape((self.numNodes, 1))     if isinstance(self.parameters['p_extern'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['p_extern'], shape=(self.numNodes,1))

        # Testing-related parameters:
        self.beta_D         = (numpy.array(self.parameters['beta_D']).reshape((self.numNodes, 1))  if isinstance(self.parameters['beta_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta_D'], shape=(self.numNodes,1))) if self.parameters['beta_D'] is not None else self.beta
        self.sigma_D        = (numpy.array(self.parameters['sigma_D']).reshape((self.numNodes, 1)) if isinstance(self.parameters['sigma_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma_D'], shape=(self.numNodes,1))) if self.parameters['sigma_D'] is not None else self.sigma
        self.gamma_D        = (numpy.array(self.parameters['gamma_D']).reshape((self.numNodes, 1)) if isinstance(self.parameters['gamma_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_D'], shape=(self.numNodes,1))) if self.parameters['gamma_D'] is not None else self.gamma
        self.mu_D           = (numpy.array(self.parameters['mu_D']).reshape((self.numNodes, 1))    if isinstance(self.parameters['mu_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_D'], shape=(self.numNodes,1))) if self.parameters['mu_D'] is not None else self.mu_I
        self.theta_E        = numpy.array(self.parameters['theta_E']).reshape((self.numNodes, 1))  if isinstance(self.parameters['theta_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_E'], shape=(self.numNodes,1))
        self.theta_I        = numpy.array(self.parameters['theta_I']).reshape((self.numNodes, 1))  if isinstance(self.parameters['theta_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_I'], shape=(self.numNodes,1))
        self.phi_E          = numpy.array(self.parameters['phi_E']).reshape((self.numNodes, 1))    if isinstance(self.parameters['phi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_E'], shape=(self.numNodes,1))
        self.phi_I          = numpy.array(self.parameters['phi_I']).reshape((self.numNodes, 1))    if isinstance(self.parameters['phi_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_I'], shape=(self.numNodes,1))
        self.psi_E          = numpy.array(self.parameters['psi_E']).reshape((self.numNodes, 1))    if isinstance(self.parameters['psi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_E'], shape=(self.numNodes,1))
        self.psi_I          = numpy.array(self.parameters['psi_I']).reshape((self.numNodes, 1))    if isinstance(self.parameters['psi_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_I'], shape=(self.numNodes,1))
        self.q              = numpy.array(self.parameters['q']).reshape((self.numNodes, 1))        if isinstance(self.parameters['q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['q'], shape=(self.numNodes,1))
        self.min_time = numpy.array(self.parameters['min_time']).reshape((self.numNodes, 1)) if isinstance(self.parameters['min_time'], (
        list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['min_time'], shape=(self.numNodes, 1))
        
        
        self.p_periodic = self.parameters['p_periodic'] 
        
        #numpy.array(self.parameters['p_periodic']).reshape((self.numNodes, 1)) if isinstance(
        #    self.parameters['p_periodic'], (list, numpy.ndarray)) else self.parameters['p_periodic'] 
        
        #  numpy.full(fill_value=self.parameters['p_periodic'], shape=(self.numNodes, 1))
        
        self.period = self.parameters['period']
        self.batch = self.parameters['batch']
        self.count_non_random = self.parameters['count_non_random']
        self.test_recovered = self.parameters['test_recovered']



        #Local transmission parameters:
        if(self.parameters['beta_local'] is not None):
            if(isinstance(self.parameters['beta_local'], (list, numpy.ndarray))):
                if(isinstance(self.parameters['beta_local'], list)):
                    self.beta_local = numpy.array(self.parameters['beta_local'])
                else: # is numpy.ndarray
                    self.beta_local = self.parameters['beta_local']
                if(self.beta_local.ndim == 1):
                    self.beta_local.reshape((self.numNodes, 1))
                elif(self.beta_local.ndim == 2):
                    self.beta_local.reshape((self.numNodes, self.numNodes))
            else:
                self.beta_local = numpy.full_like(self.beta, fill_value=self.parameters['beta_local'])
        else:
            self.beta_local = self.beta
        #----------------------------------------
        if(self.parameters['beta_D_local'] is not None):
            if(isinstance(self.parameters['beta_D_local'], (list, numpy.ndarray))):
                if(isinstance(self.parameters['beta_D_local'], list)):
                    self.beta_D_local = numpy.array(self.parameters['beta_D_local'])
                else: # is numpy.ndarray
                    self.beta_D_local = self.parameters['beta_D_local']
                if(self.beta_D_local.ndim == 1):
                    self.beta_D_local.reshape((self.numNodes, 1))
                elif(self.beta_D_local.ndim == 2):
                    self.beta_D_local.reshape((self.numNodes, self.numNodes))
            else:
                self.beta_D_local = numpy.full_like(self.beta_D, fill_value=self.parameters['beta_D_local'])
        else:
            self.beta_D_local = self.beta_D
        
        # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
        if(self.beta_local.ndim == 1):
            self.A_beta     = scipy.sparse.csr_matrix.multiply(self.A, numpy.tile(self.beta_local, (1,self.numNodes))).tocsr()
        elif(self.beta_local.ndim == 2):
            self.A_beta     = scipy.sparse.csr_matrix.multiply(self.A, self.beta_local).tocsr()
        # Pre-multiply beta_D values by the quarantine adjacency matrix ("transmission weight connections")
        if(self.beta_D_local.ndim == 1):
            self.A_Q_beta_D = scipy.sparse.csr_matrix.multiply(self.A_Q, numpy.tile(self.beta_D_local, (1,self.numNodes))).tocsr()
        elif(self.beta_D_local.ndim == 2):
            self.A_Q_beta_D = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_D_local).tocsr()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update scenario flags:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.update_scenario_flags()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_G(self, new_G):
        self.G = new_G
        # Adjacency matrix:
        if type(new_G)==numpy.ndarray:
            self.A = scipy.sparse.csr_matrix(new_G)
        elif type(new_G)==networkx.classes.graph.Graph:
            self.A = networkx.adj_matrix(new_G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes   = int(self.A.shape[1])
        self.degree     = numpy.asarray(self.node_degrees(self.A)).astype(float)

        return

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_Q(self, new_Q):
        self.Q = new_Q
        # Quarantine Adjacency matrix:
        if type(new_Q)==numpy.ndarray:
            self.A_Q = scipy.sparse.csr_matrix(new_Q)
        elif type(new_Q)==networkx.classes.graph.Graph:
            self.A_Q = networkx.adj_matrix(new_Q) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes_Q   = int(self.A_Q.shape[1])
        self.degree_Q     = numpy.asarray(self.node_degrees(self.A_Q)).astype(float)

        assert(self.numNodes == self.numNodes_Q), "The normal and quarantine adjacency graphs must be of the same size."

        return

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_scenario_flags(self):
        self.testing_scenario   = ( (numpy.any(self.psi_I) and (numpy.any(self.theta_I) or numpy.any(self.phi_I)))  
                                    or (numpy.any(self.psi_E) and (numpy.any(self.theta_E) or numpy.any(self.phi_E))) or numpy.any(self.p_periodic) )
        self.tracing_scenario   = ( (numpy.any(self.psi_E) and numpy.any(self.phi_E)) 
                                    or (numpy.any(self.psi_I) and numpy.any(self.phi_I)) )
        self.vitality_scenario  = (numpy.any(self.mu_0) and numpy.any(self.nu))
        self.resusceptibility_scenario  = (numpy.any(self.xi))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_infections(self, t_idx=None):
        if(t_idx is None):
            return (self.numE[:] + self.numI[:] + self.numD_E[:] + self.numD_I[:])            
        else:
            return (self.numE[t_idx] + self.numI[t_idx] + self.numD_E[t_idx] + self.numD_I[t_idx])          


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        
    def calc_propensities(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication
        transmissionTerms_I = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numI[self.tidx]) 
            and numpy.any(self.beta!=0)):
            transmissionTerms_I = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A_beta, self.X==self.I) )

        transmissionTerms_DI = numpy.zeros(shape=(self.numNodes,1))
        if(self.testing_scenario 
            and numpy.any(self.numD_I[self.tidx])
            and numpy.any(self.beta_D)):
            transmissionTerms_DI = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A_Q_beta_D, self.X==self.D_I) )

        numContacts_D = numpy.zeros(shape=(self.numNodes,1))
        if(self.tracing_scenario 
            and (numpy.any(self.numD_E[self.tidx]) or numpy.any(self.numD_I[self.tidx]))):
            numContacts_D = numpy.asarray( scipy.sparse.csr_matrix.dot( self.A, ((self.X==self.D_E)|(self.X==self.D_I)) ) )
                                            

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #  BOAZ: additions
        #NS - negative S
        #NE - negative E
        #NI - negative I



        propensities_StoE   = (self.p_extern +  self.p*((self.beta*self.numI[self.tidx] + self.q*self.beta_D*self.numD_I[self.tidx])/self.N[self.tidx])
                                + (1-self.p)*numpy.divide((transmissionTerms_I + transmissionTerms_DI), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0)
                              )*(self.X==self.S)

        propensities_EtoI   = self.sigma*(self.X==self.E)

        propensities_ItoR   = self.gamma*(self.X==self.I)

        propensities_ItoF   = self.mu_I*(self.X==self.I)

        # propensities_EtoDE  = ( self.theta_E + numpy.divide((self.phi_E*numContacts_D), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0) )*self.psi_E*(self.X==self.E)

        can_test = self.last_tested + self.min_time <= self.t

        propensities_StoNS = (self.phi_E * numContacts_D) * (self.X == self.S) * can_test

        propensities_EtoDE  = (self.theta_E + self.phi_E*numContacts_D)*self.psi_E*(self.X==self.E) * can_test



        # propensities_ItoDI  = ( self.theta_I + numpy.divide((self.phi_I*numContacts_D), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0) )*self.psi_I*(self.X==self.I)
        propensities_ItoDI  = (self.theta_I + self.phi_I*numContacts_D)*self.psi_I*(self.X==self.I) * can_test

        propensities_DEtoDI = self.sigma_D*(self.X==self.D_E)

        propensities_DItoR  = self.gamma_D*(self.X==self.D_I)

        propensities_DItoF  = self.mu_D*(self.X==self.D_I)

        propensities_RtoS   = self.xi*(self.X==self.R)

        propensities__toS   = self.nu*(self.X!=self.F)

        propensities = numpy.hstack([propensities_StoE, propensities_EtoI, 
                                     propensities_ItoR, propensities_ItoF, 
                                     propensities_EtoDE, propensities_ItoDI, propensities_DEtoDI, 
                                     propensities_DItoR, propensities_DItoF,
                                     propensities_RtoS, propensities__toS , propensities_StoNS])

        columns = ['StoE', 'EtoI', 'ItoR', 'ItoF', 'EtoDE', 'ItoDI', 'DEtoDI', 'DItoR', 'DItoF', 'RtoS', '_toS' , 'StoNS']

        return propensities, columns


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

    def increase_data_series_length(self):
        self.tseries= numpy.pad(self.tseries, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numS   = numpy.pad(self.numS, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numE   = numpy.pad(self.numE, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numI   = numpy.pad(self.numI, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_E = numpy.pad(self.numD_E, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_I = numpy.pad(self.numD_I, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numR   = numpy.pad(self.numR, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numF   = numpy.pad(self.numF, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.N      = numpy.pad(self.N, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numTested = numpy.pad(self.numTested, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numPositive = numpy.pad(self.numPositive, [(0, 5 * self.numNodes)], mode='constant', constant_values=0)

        if(self.store_Xseries):
            self.Xseries = numpy.pad(self.Xseries, [(0, 5*self.numNodes), (0,0)], mode='constant', constant_values=0)

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']     = numpy.pad(self.nodeGroupData[groupName]['numS'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numE']     = numpy.pad(self.nodeGroupData[groupName]['numE'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI']     = numpy.pad(self.nodeGroupData[groupName]['numI'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD_E']   = numpy.pad(self.nodeGroupData[groupName]['numD_E'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD_I']   = numpy.pad(self.nodeGroupData[groupName]['numD_I'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numR']     = numpy.pad(self.nodeGroupData[groupName]['numR'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numF']     = numpy.pad(self.nodeGroupData[groupName]['numF'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['N']        = numpy.pad(self.nodeGroupData[groupName]['N'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numTested'] = numpy.pad(self.nodeGroupData[groupName]['numTested'],
                                                               [(0, 5 * self.numNodes)], mode='constant',
                                                               constant_values=0)

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

    def finalize_data_series(self):
        self.tseries= numpy.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS   = numpy.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE   = numpy.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI   = numpy.array(self.numI, dtype=float)[:self.tidx+1]
        self.numD_E = numpy.array(self.numD_E, dtype=float)[:self.tidx+1]
        self.numD_I = numpy.array(self.numD_I, dtype=float)[:self.tidx+1]
        self.numR   = numpy.array(self.numR, dtype=float)[:self.tidx+1]
        self.numF   = numpy.array(self.numF, dtype=float)[:self.tidx+1]
        self.N      = numpy.array(self.N, dtype=float)[:self.tidx+1]
        self.numTested      = numpy.array(self.numTested, dtype=float)[:self.tidx+1]
        self.numPositive = numpy.array(self.numPositive, dtype=float)[:self.tidx + 1]

        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']    = numpy.array(self.nodeGroupData[groupName]['numS'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numE']    = numpy.array(self.nodeGroupData[groupName]['numE'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI']    = numpy.array(self.nodeGroupData[groupName]['numI'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD_E']  = numpy.array(self.nodeGroupData[groupName]['numD_E'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD_I']  = numpy.array(self.nodeGroupData[groupName]['numD_I'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numR']    = numpy.array(self.nodeGroupData[groupName]['numR'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numF']    = numpy.array(self.nodeGroupData[groupName]['numF'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['N']       = numpy.array(self.nodeGroupData[groupName]['N'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numTested'] = numpy.array(self.nodeGroupData[groupName]['numTested'], dtype=float)[:self.tidx + 1]

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def periodic_test(self, p):
        if (self.tidx >= len(self.tseries) - 1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()
        self.tidx+= 1

        if self.batch and self.nodeGroupData:
            if not isinstance(p,dict):
                s = f"p_periodic should be dict but is {p}"
                print(s)
                raise Exception(s)
            test = numpy.zeros((self.numNodes,1))
            for groupName , prop in self.nodeGroupData.items():
                a = self.nodeToTest[groupName]
                n = len(prop['nodes'])
                q = p[groupName]
                b = int(a + q*n)
                if b>n:
                    nodes = list(prop['nodes'][a:n])+list(prop['nodes'][0:b-n])
                    self.nodeToTest[groupName] = b-n
                else
                    nodes = list(prop['nodes'][a:b])
                    self.nodeToTest[groupName] = b - n
                test += numpy.isin(range(self.numNodes), nodes).reshape((self.numNodes, 1))
        elif self.batch:
            test = numpy.zeros((self.numNodes,1) , dtype = bool)
            if not isinstance(p,(float,int)):
                s = f"p_periodic should be float but is {p}"
                print(f"p_periodic should be float but is {p}")
                raise Exception(s)
                
            if isinstance(p, (list, numpy.ndarray)):
                p = numpy.average(p)

            b = int(self.nodeToTest + p*self.numNodes)
            #print(f"Periodic testing p = {p}  nodeToTest = {self.nodeToTest} b ={b}")
            if b < self.numNodes:
                test[self.nodeToTest:b] = True
                self.nodeToTest = b
            else:
                b = b - self.numNodes
                test[self.nodeToTest:self.numNodes] = True
                test[0:b] = True
                self.nodeToTest = b


        else:
            r = numpy.random.rand(self.numNodes,1)
            #print(f"p:  {p.shape}, r: {r.shape}")
            test = r <= p

        can_test = self.last_tested + self.min_time  <= self.t
        if self.test_recovered:
            test = test * can_test * ((self.X == self.E) + (self.X == self.I) + (self.X == self.S) + (self.X == self.R))
        else:
            test = test * can_test * ((self.X == self.E) + (self.X == self.I) + (self.X == self.S))
        #print(f"t = {self.t}, testing {numpy.sum(test)} people")
        self.numTested[self.tidx] = numpy.sum(test)
        self.last_tested[test] = self.t

        # change to E that are positive
        q = numpy.random.rand(self.numNodes,1)
        positive = test * (self.X== self.E) * (q <= self.psi_E)
        #print(f"X:  {self.X.shape}, test: {test.shape} , q: {q.shape} psi_E: {self.psi_E.shape}, positive: {positive.shape}")
        self.X[positive] = self.D_E
        self.numPositive[self.tidx] =numpy.sum(positive)


        # change to I that are positive
        positive = test * (self.X == self.I) * (q <= self.psi_I)
        self.X[positive] = self.D_I
        self.numPositive[self.tidx] += numpy.sum(positive)

        self.tseries[self.tidx]  = self.t
        self.numS[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numD_E[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_E), a_min=0, a_max=self.numNodes)
        self.numD_I[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_I), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.N[self.tidx]        = numpy.clip((self.numS[self.tidx] + self.numE[self.tidx] + self.numI[self.tidx] + self.numD_E[self.tidx] + self.numD_I[self.tidx] + self.numR[self.tidx]), a_min=0, a_max=self.numNodes)


        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numD_E'][self.tidx]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_E)
                self.nodeGroupData[groupName]['numD_I'][self.tidx]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_I)
                self.nodeGroupData[groupName]['numR'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numTested'][self.tidx] = nnumpy.sum(test* self.nodeGroupData[groupName]['mask'])
                self.nodeGroupData[groupName]['N'][self.tidx]       = numpy.clip((self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numD_E'][0] + self.nodeGroupData[groupName]['numD_I'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)


    def run_iteration(self):


        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()
        
        
        if self.t < self.wait_until_t:
            if int(self.t) < int(self.wait_until_t):
                self.t = int(self.t)+1
            else:
                self.t = self.wait_until_t + 0.01 / self.numNodes
            transitionType = None
        else:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 1. Generate 2 random numbers uniformly distributed in (0,1)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            r1 = numpy.random.rand()
            r2 = numpy.random.rand()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 2. Calculate propensities
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities, transitionTypes = self.calc_propensities()

            # Terminate when probability of all events is 0:
            if(propensities.sum() <= 0.0):            
                self.finalize_data_series()
                return False

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3. Calculate alpha
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities_flat   = propensities.ravel(order='F')
            cumsum              = propensities_flat.cumsum()
            alpha               = propensities_flat.sum()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 4. Compute the time until the next event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tau = (1/alpha)*numpy.log(float(1/r1))
            if tau < 0.0001/self.numNodes:
                p_extern = numpy.mean(self.p_extern)  if isinstance(self.p_extern, (list, numpy.ndarray)) else self.p_extern
                if self.small_step == True:
                    # If step is very small we make a bigger time delta
                    print(f"At {self.t}, tau={tau}, p_extern = {p_extern} ")
                    tau = 0.01/self.numNodes
                    self.small_step = False
                else:
                    self.small_step = True
            else:
                self.small_step = False
                
            self.wait_until_t = self.t + tau
            if int(self.wait_until_t) > int(self.t):
                self.t = int(self.t)+1
            else:
                self.t = self.wait_until_t

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 5. Compute which event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            transitionIdx   = numpy.searchsorted(cumsum,r2*alpha)
            transitionNode  = transitionIdx % self.numNodes
            transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]
            




            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 6. Update node states and data series
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
            self.X[transitionNode] = self.transitions[transitionType]['newState']

        self.tidx += 1

        self.tseries[self.tidx]  = self.t
        self.numS[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numD_E[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_E), a_min=0, a_max=self.numNodes)
        self.numD_I[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_I), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.N[self.tidx]        = numpy.clip((self.numS[self.tidx] + self.numE[self.tidx] + self.numI[self.tidx] + self.numD_E[self.tidx] + self.numD_I[self.tidx] + self.numR[self.tidx]), a_min=0, a_max=self.numNodes)
        
        if transitionType in ['EtoDE', 'ItoDI', 'StoNS']:
            if self.count_non_random:
                self.numTested[self.tidx] = 1
                self.last_tested[transitionNode] = self.t
        if transitionType in ['EtoDE', 'ItoDI']:
            self.numPositive[self.tidx] = 1





        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numD_E'][self.tidx]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_E)
                self.nodeGroupData[groupName]['numD_I'][self.tidx]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_I)
                self.nodeGroupData[groupName]['numR'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['N'][self.tidx]       = numpy.clip((self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numD_E'][0] + self.nodeGroupData[groupName]['numD_I'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)
                self.nodeGroupData[groupName]['numTested'][self.tidx] = self.numTested[self.tidx]*self.nodeGroupData[groupName]['mask'][transitionNode]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infectious and num exposed is 0: (boaz: dropped this condition since we have external infections)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax): # or (self.numI[self.tidx]<1 and self.numE[self.tidx]<1 and self.numD_E[self.tidx]<1 and self.numD_I[self.tidx]<1)):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




# BOAZ: added stopping argument to give a stopping condition
# policy function to modify parameters adaptively every policyInterval
# runTillEnd to force the simulation to run till end even if no disease is present
    def run(self, T, checkpoints=None, print_interval=10, verbose='t',runTillEnd = True , stopping = lambda s: False, policyInterval = 1 , type="", variant=""):
        if(T>0):
            self.tmax += T
        else:
            return False
        
        
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            for chkpt_param, chkpt_values in checkpoints.items():
                assert(isinstance(chkpt_values, (list, numpy.ndarray)) and len(chkpt_values)==numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
            checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if(checkpointIdx >= numCheckpoints):
                # We are out of checkpoints, stop checking them:
                checkpoints = None 
            else:
                checkpointTime = checkpoints['t'][checkpointIdx]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print_reset = True
        running     = True
        lastPolicyInterval = 0
        lastTestInterval = 0
        if verbose:
            print("Starting simulation")
        step_counter = 0
        max_steps = self.numNodes*100 + T*100
        
        while (step_counter < max_steps) and  (running or runTillEnd):
            step_counter += 1
            if verbose and (step_counter % 1000 == 0):
                print(f"{self.t}/{self.tidx} ({self.numD_I[self.tidx] + self.numD_E[self.tidx] +self.numI[self.tidx] + self.numE[self.tidx]}) ",end="",flush=True)
            running = self.run_iteration()
            if self.t >= self.tmax: 
                runTillEnd = False
                running = False
            temp = numpy.mean(self.p_extern)  if isinstance(self.p_extern, (list, numpy.ndarray)) else self.p_extern
            if step_counter >= max_steps:
                print("Step counter more than maximum - stopping at {self.t} ({variant}, p_extern =  {p_extern})")

            if self.t > 0 and  temp < 0.0001/self.numNodes and not (self.numD_I[self.tidx] + self.numD_E[self.tidx] +self.numI[self.tidx] + self.numE[self.tidx]  ):
                if verbose:
                    print(f"t = {self.t} , mean p_extern = {temp} and no infected people: stopping")
                running = False
                runTillEnd = False
                
            

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            if(checkpoints):
                if(self.t >= checkpointTime):
                    if(verbose is not False):
                        print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    if('G' in list(checkpoints.keys())):
                        self.update_G(checkpoints['G'][checkpointIdx])
                    if('Q' in list(checkpoints.keys())):
                        self.update_Q(checkpoints['Q'][checkpointIdx])
                    for param in list(self.parameters.keys()):
                        if(param in list(checkpoints.keys())):
                            self.parameters.update({param: checkpoints[param][checkpointIdx]})
                    # Update parameter data structures and scenario flags:
                    self.update_parameters()
                    # Update the next checkpoint time:
                    checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if(checkpointIdx >= numCheckpoints):
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None 
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #print(f"t={self.t}, period ={self.period}")
            if numpy.any(self.p_periodic) \
                    and (self.period>0) \
                    and  ( int(self.t / self.period) > lastTestInterval):
                if verbose:
                    print(f"periodic testing t={self.t}")
                self.periodic_test(self.p_periodic)
                lastTestInterval = int(self.t / self.period)


            if self.has_policy and (not policyInterval  or (int(self.t / policyInterval)> lastPolicyInterval)):
                lastPolicyInterval  = int(self.t / policyInterval)
                if(verbose):
                    print(f"t={self.t}, Applying policy")
                self.policy()
                # self.update_parameters()

            if self.numD_E[self.tidx] + self.numD_I[self.tidx]:
                if not self.time_detected:
                    self.time_detected = self.tidx
            
            if stopping == "1st":
                if self.numD_I[self.tidx] + self.numD_E[self.tidx]:
                    self.finalize_data_series()
                    running = False
                    runTillEnd = False
            elif stopping and stopping(self):
                self.finalize_data_series()
                running = False
                runTillEnd = False

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(print_interval):
                if(print_reset and (int(self.t) % print_interval == 0)):
                    if(verbose=="t"):
                        print("t = %.2f" % self.t)
                    if(verbose==True):
                        print("t = %.2f" % self.t)
                        print("\t S   = " + str(self.numS[self.tidx]))
                        print("\t E   = " + str(self.numE[self.tidx]))
                        print("\t I   = " + str(self.numI[self.tidx]))
                        print("\t D_E = " + str(self.numD_E[self.tidx]))
                        print("\t D_I = " + str(self.numD_I[self.tidx]))
                        print("\t R   = " + str(self.numR[self.tidx]))
                        print("\t F   = " + str(self.numF[self.tidx]))
                    print_reset = False
                elif(not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True

        
        # end of while loop
        if not self.time_detected:
            self.time_detected = self.tidx
        
        if (verbose):
            print(f"Finished execution at {self.t}")
            print(f"percentage of population tested / day: {100*numpy.sum(self.numTested)/(self.numNodes*self.t):.3f}%")
    
        totscale = 1
        finscale = 1 # 100.0/(self.numNodes)
        log = dict(self.init_parameters)
        temp = numpy.roll(self.tseries,1)
        temp[0] = 0
        lengths = self.tseries - temp
        log.update(
            { "type": type,
             "variant": variant,
              "checkpoints": checkpoints,
             "policy": self.has_policy ,
             "policy interval": policyInterval,
             "stopping": stopping,
             "t":  self.t,
             "totS": numpy.sum(self.numS * lengths),
             "totE": numpy.sum(self.numE * lengths),
             "totI": numpy.sum(self.numI * lengths),
             "totD_E": numpy.sum(self.numD_E * lengths),
             "totD_I" : numpy.sum(self.numD_I* lengths),
             "totE_tillFirst": numpy.sum(self.numE[:self.time_detected+1] * lengths[:self.time_detected+1]),
             "totI_tillFirst": numpy.sum(self.numI[:self.time_detected+1] * lengths[:self.time_detected+1]),
             "totR"   : numpy.sum(self.numR * lengths),
             "tit"   : numpy.sum(self.numF * lengths),
             "totTests": numpy.sum(self.numTested),
             "totTests1st": numpy.sum(self.numTested[:self.time_detected+1]),
             "meanTests1st" : numpy.sum(self.numTested[:self.time_detected+1])/self.tseries[self.time_detected] if self.time_detected else 0,
             "totPositive": numpy.sum(self.numPositive),
             "finS": self.numS[self.tidx]*finscale,
             "finE": self.numE[self.tidx]*finscale,
             "finI": self.numI[self.tidx]*finscale,
             "finD_E": self.numD_E[self.tidx]*finscale,
             "finD_I": self.numD_I[self.tidx]*finscale,
             "finR": self.numR[self.tidx]*finscale,
             "finF": self.numF[self.tidx]*finscale,
             "note": f"Finals scaled by {finscale:.5f}. Averages per time period",
             "time1st" : self.tseries[self.time_detected]
             })

        if (self.nodeGroupData):
            for groupName in self.nodeGroupData:
                log.update({
                    f"{groupName}_totS": numpy.sum(self.nodeGroupData[groupName]['numS'] * lengths),
                    f"{groupName}_totE": numpy.sum(self.nodeGroupData[groupName]['numE'] * lengths),
                    f"{groupName}_totI": numpy.sum(self.nodeGroupData[groupName]['numI'] * lengths),
                    f"{groupName}_totD_E": numpy.sum(self.nodeGroupData[groupName]['numD_E'] * lengths),
                    f"{groupName}_totD_I": numpy.sum(self.nodeGroupData[groupName]['numD_I'] * lengths),
                    f"{groupName}_totE_tillFirst": numpy.sum(self.nodeGroupData[groupName]['numE'][:self.time_detected + 1] * lengths[:self.time_detected + 1]),
                    f"{groupName}_totI_tillFirst": numpy.sum(self.nodeGroupData[groupName]['numI'][:self.time_detected + 1] * lengths[:self.time_detected + 1]),
                    f"{groupName}_totR": numpy.sum(self.nodeGroupData[groupName]['numR'] * lengths),
                    f"{groupName}_totTests": numpy.sum(self.nodeGroupData[groupName]['numTested']),
                    f"{groupName}_totTests1st": numpy.sum(self.nodeGroupData[groupName]['numTested'][:self.time_detected + 1]),
                    f"{groupName}_meanTests1st": numpy.sum(self.nodeGroupData[groupName]['numTested'][:self.time_detected + 1]) / self.tseries[
                        self.time_detected] if self.time_detected else 0,
                    f"{groupName}_finS": self.nodeGroupData[groupName]['numS'][self.tidx] * finscale,
                    f"{groupName}_finE": self.nodeGroupData[groupName]['numE'][self.tidx] * finscale,
                    f"{groupName}_finI": self.nodeGroupData[groupName]['numI'][self.tidx] * finscale,
                    f"{groupName}_finD_E": self.nodeGroupData[groupName]['numD_E'][self.tidx] * finscale,
                    f"{groupName}_finD_I": self.nodeGroupData[groupName]['numD_I'][self.tidx] * finscale,
                    f"{groupName}_finR": self.nodeGroupData[groupName]['numR'][self.tidx] * finscale,
                    f"{groupName}_finF": self.nodeGroupData[groupName]['numF'][self.tidx] * finscale,

                })
                log.update({
                f"{groupName}_undetected1st":  self.nodeGroupData[groupName]['numE'][self.time_detected] + self.nodeGroupData[groupName]['numI'][self.time_detected],
                f"{groupName}_infected1st":    self.nodeGroupData[groupName]['numE'][self.time_detected] + self.nodeGroupData[groupName]['numI'][self.time_detected] + self.nodeGroupData[groupName]['numD_E'][self.time_detected] + self.nodeGroupData[groupName]['numD_I'][self.time_detected],
                f"{groupName}_totUndetected1st": log[f"{groupName}_totE_tillFirst"] + log[f"{groupName}_totI_tillFirst"],
                f"{groupName}_meanUndetected1st": (log[f"{groupName}_totE_tillFirst"] + log[f"{groupName}_totI_tillFirst"])/self.tseries[self.time_detected]  if self.time_detected else 0
                })

        time1st = self.tseries[self.time_detected]
        
        log.update({
                "totInfected": log["totE"]+log["totI"]+log["totD_E"]+log["totD_I"],
                "maxInfected": numpy.max(self.numE + self.numI + self.numD_E + self.numD_I),
                "finInfected": log["finE"] + log["finI"] + log["finD_E"] + log["finD_I"],
                "totUndetected": log["totE"]+log["totI"],
                "meanUndetectedInfectiousDays" : log["totI"]/self.t,
                "meanUndetected": (log["totE"] + log["totI"])/self.t,
                "undetected1st":  self.numE[self.time_detected] + self.numI[self.time_detected],
                "infected1st":    self.numE[self.time_detected] + self.numI[self.time_detected] + self.numD_E[self.time_detected] + self.numD_I[self.time_detected],
                "totUndetected1st": log["totE_tillFirst"] + log["totI_tillFirst"],
                "meanUndetected1st": (log["totE_tillFirst"] + log["totI_tillFirst"])/self.tseries[self.time_detected]  if self.time_detected else 0,
                "meanTests": log["totTests"]/self.t,
                "finUndetected": log["finE"]+log["finI"],
                "overall_infected": self.numNodes - log["finS"]})

        # compute baseline risk if the workplace was closed.

        mean_p_extern = numpy.mean(self.p_extern)  if isinstance(self.p_extern, (list, numpy.ndarray)) else self.p_extern

        gamma =   numpy.mean(self.gamma)  if isinstance(self.gamma, (list, numpy.ndarray)) else self.gamma
        base_risk = mean_p_extern / gamma

        log["excessRisk"] = 100*(log["totI"]/(self.t*self.numNodes) - base_risk)/base_risk if base_risk else 0


        return log


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def plot(self, ax=None,  plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                            plot_D_E='line', plot_D_I='line', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True, plot_tested = True):

        import matplotlib.pyplot as pyplot

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(not ax):
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Fseries     = self.numF/self.numNodes if plot_percentages else self.numF
        Eseries     = self.numE/self.numNodes if plot_percentages else self.numE
        Dseries     = (self.numD_E+self.numD_I)/self.numNodes if plot_percentages else (self.numD_E+self.numD_I)
        D_Eseries   = self.numD_E/self.numNodes if plot_percentages else self.numD_E
        D_Iseries   = self.numD_I/self.numNodes if plot_percentages else self.numD_I
        Iseries     = self.numI/self.numNodes if plot_percentages else self.numI
        Rseries     = self.numR/self.numNodes if plot_percentages else self.numR
        Sseries     = self.numS/self.numNodes if plot_percentages else self.numS 
        Testseries   = self.numTested/ self.numNodes if plot_percentages  else self.numTested

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
        # Draw tested fraction
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (plot_tested):
            # average over 3 days
            #temp = [0] * (int(self.t/3) +2)
            #for id in range(self.tidx):
            #    temp[int(self.tseries[id]/3)] += Testseries[id]
            #for id in range(self.tidx):
            #    Testseries[id] = temp[int(self.tseries[id]/3)]/3
            #ax.plot(self.tseries, Testseries,color='grey', linestyle='--', label ='tested', zorder=1)
            ax.fill_between(numpy.ma.masked_where(Testseries<=0, self.tseries), numpy.ma.masked_where(Testseries<=0, Testseries), color='grey', label = 'tested', alpha=0.4, zorder=4)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:      
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(dashed_reference_results):
            dashedReference_tseries  = dashed_reference_results.tseries[::int(self.numNodes/100)]
            dashedReference_IDEstack = (dashed_reference_results.numI + dashed_reference_results.numD_I + dashed_reference_results.numD_E + dashed_reference_results.numE)[::int(self.numNodes/100)] / (self.numNodes if plot_percentages else 1)
            ax.plot(dashedReference_tseries, dashedReference_IDEstack, color='#E0E0E0', linestyle='--', label='$I+D+E$ ('+dashed_reference_label+')', zorder=0)
        if(shaded_reference_results):
            shadedReference_tseries  = shaded_reference_results.tseries
            shadedReference_IDEstack = (shaded_reference_results.numI + shaded_reference_results.numD_I + shaded_reference_results.numD_E + shaded_reference_results.numE) / (self.numNodes if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries, shadedReference_IDEstack, 0, color='#EFEFEF', label='$I+D+E$ ('+shaded_reference_label+')', zorder=0)
            ax.plot(shaded_reference_results.tseries, shadedReference_IDEstack, color='#E0E0E0', zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = numpy.zeros_like(self.tseries)
        if(any(Fseries) and plot_F=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries), topstack, color=color_F, alpha=0.5, label='$F$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries),           color=color_F, zorder=3)
            topstack = topstack+Fseries
        if(any(Eseries) and plot_E=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='$E$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries),           color=color_E, zorder=3)
            topstack = topstack+Eseries
        if(combine_D and plot_D_E=='stacked' and plot_D_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries), topstack, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries),           color=color_D_E, zorder=3)
            topstack = topstack+Dseries
        else:
            if(any(D_Eseries) and plot_D_E=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries), topstack, color=color_D_E, alpha=0.5, label='$D_E$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries),           color=color_D_E, zorder=3)
                topstack = topstack+D_Eseries
            if(any(D_Iseries) and plot_D_I=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, topstack+D_Iseries), topstack, color=color_D_I, alpha=0.5, label='$D_I$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, topstack+D_Iseries),           color=color_D_I, zorder=3)
                topstack = topstack+D_Iseries
        if(any(Iseries) and plot_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries),           color=color_I, zorder=3)
            topstack = topstack+Iseries
        if(any(Rseries) and plot_R=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries),           color=color_R, zorder=3)
            topstack = topstack+Rseries
        if(any(Sseries) and plot_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries),           color=color_S, zorder=3)
            topstack = topstack+Sseries
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(any(Fseries) and plot_F=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), 0, color=color_F, alpha=0.5, label='$F$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries),    color=color_F, zorder=5)
        if(any(Eseries) and plot_E=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), 0, color=color_E, alpha=0.5, label='$E$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries),    color=color_E, zorder=5)
        if(combine_D and (any(Dseries) and plot_D_E=='shaded' and plot_D_I=='shaded')):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), 0, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries),    color=color_D_E, zorder=5)
        else:
            if(any(D_Eseries) and plot_D_E=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), 0, color=color_D_E, alpha=0.5, label='$D_E$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries),    color=color_D_E, zorder=5)
            if(any(D_Iseries) and plot_D_I=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries), 0, color=color_D_I, alpha=0.5, label='$D_I$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries),    color=color_D_I, zorder=5)
        if(any(Iseries) and plot_I=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), 0, color=color_I, alpha=0.5, label='$I$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries),    color=color_I, zorder=5)
        if(any(Sseries) and plot_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), 0, color=color_S, alpha=0.5, label='$S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries),    color=color_S, zorder=5)
        if(any(Rseries) and plot_R=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), 0, color=color_R, alpha=0.5, label='$R$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries),    color=color_R, zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        if(any(Fseries) and plot_F=='line'):
            ax.plot(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), color=color_F, label='$F$', zorder=6)
        if(any(Eseries) and plot_E=='line'):
            ax.plot(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='$E$', zorder=6)
        if(combine_D and (any(Dseries) and plot_D_E=='line' and plot_D_I=='line')):
            ax.plot(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), color=color_D_E, label='$D_{all}$', zorder=6)
        else:
            if(any(D_Eseries) and plot_D_E=='line'):
                ax.plot(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), color=color_D_E, label='$D_E$', zorder=6)
            if(any(D_Iseries) and plot_D_I=='line'):
                ax.plot(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries), color=color_D_I, label='$D_I$', zorder=6)
        if(any(Iseries) and plot_I=='line'):
            ax.plot(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='$I$', zorder=6)
        if(any(Sseries) and plot_S=='line'):
            ax.plot(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)
        if(any(Rseries) and plot_R=='line'):
            ax.plot(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(len(vlines)>0 and len(vline_colors)==0):
            vline_colors = ['gray']*len(vlines)
        if(len(vlines)>0 and len(vline_labels)==0):
            vline_labels = [None]*len(vlines)
        if(len(vlines)>0 and len(vline_styles)==0):
            vline_styles = [':']*len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
            if(vline_x is not None):
                ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, alpha=1, label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('days')
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
        if(title):
            ax.set_title(title, size=12)
        if(side_title):
            ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')
       
        return ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self, plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                        plot_D_E='line', plot_D_I='line', combine_D=True,
                        color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                        color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                        dashed_reference_results=None, dashed_reference_label='reference', 
                        shaded_reference_results=None, shaded_reference_label='reference', 
                        vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                        ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                        figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self, plot_S=False, plot_E='stacked', plot_I='stacked',plot_R=False, plot_F=False,
                            plot_D_E='stacked', plot_D_I='stacked', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                            figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels, 
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a custom method for generating 
# power-law-like graphs with exponential tails 
# both above and below the degree mean and  
# where the mean degree be easily down-shifted
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    # Generate a random preferential attachment power law graph as a starting point.
    # By the way this graph is constructed, it is expected to have 1 connected component.
    # Every node is added along with m=8 edges, so the min degree is m=8.
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = networkx.barabasi_albert_graph(n=n, m=m)

    # To get a graph with power-law-esque properties but without the fixed minimum degree,
    # We modify the graph by probabilistically dropping some edges from each node. 
    for node in graph:
        neighbors = list(graph[node].keys())
        quarantineEdgeNum = int( max(min(numpy.random.exponential(scale=scale, size=1), len(neighbors)), min_num_edges) )
        quarantineKeepNeighbors = numpy.random.choice(neighbors, size=quarantineEdgeNum, replace=False)
        for neighbor in neighbors:
            if(neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)
    
    return graph

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def plot_degree_distn(graph, max_degree=None, show=True, use_seaborn=True):
    import matplotlib.pyplot as pyplot
    if(use_seaborn):
        import seaborn
        seaborn.set_style('ticks')
        seaborn.despine()
    # Get a list of the node degrees:
    if type(graph)==numpy.ndarray:
        nodeDegrees = graph.sum(axis=0).reshape((graph.shape[0],1))   # sums of adj matrix cols
    elif type(graph)==networkx.classes.graph.Graph:
        nodeDegrees = [d[1] for d in graph.degree()]
    else:
        raise BaseException("Input an adjacency matrix or networkx object only.")
    # Calculate the mean degree:
    meanDegree = numpy.mean(nodeDegrees)
    # Generate a histogram of the node degrees:
    pyplot.hist(nodeDegrees, bins=range(max(nodeDegrees)), alpha=0.5, color='tab:blue', label=('mean degree = %.1f' % meanDegree))
    pyplot.xlim(0, max(nodeDegrees) if not max_degree else max_degree)
    pyplot.xlabel('degree')
    pyplot.ylabel('num nodes')
    pyplot.legend(loc='upper right')
    if(show):
        pyplot.show()



