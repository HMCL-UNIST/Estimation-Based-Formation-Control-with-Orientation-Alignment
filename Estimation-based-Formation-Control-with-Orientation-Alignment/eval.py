import numpy as np
from utils import *

class MASEval:    
    def __init__(self, args):        
        self.args = args
        self.stage_costs = []        
        self.trajs = []
        self.thetas = []
        self.thetahats = []
        self.est_trajs = []
        self.xhats = []
        self.z_trajs = []
        self.n = []
    def add_stage_cost(self,cost):
        self.stage_costs.append(cost.copy())
        
    def eval_init(self):
        plot_x_y(self.trajs,self.thetas,self.n)  
        plot_theta(self.thetas)

    def get_results(self):
        results = {}
        results['thetas'] = self.thetas
        results['trajs'] = self.trajs
        results['xhats'] = self.xhats
        results['thetahats'] = self.thetahats
        results['n'] = self.n
        return results
            
        
        