import gymnasium as gym
import pandas as pd
import numpy as np
from models import *

class SurrogateEnv:
    
    def __init__(
        self, 
        boiler_model:BoilerTimeSeriesModel, 
        chiller_model:ChillerTimeSeriesModel, 
        electrical_model:ElectricalTimeSeriesModel, 
        input_names, 
        output_names,
        demand_data:pd.DataFrame,
    ):
        
        self.boiler_model = boiler_model
        self.chiller_model = chiller_model
        self.electrical_model = electrical_model
        self.input_names = input_names
        self.output_names = output_names
        self.demand_data = demand_data
        
        self.observation_space = gym.Spaces.Box(low = 0, high = 1, shape = (26, )) #TODO
        self.action_space = gym.Spaces.Box(low = 0, high = 1, shape = (8, )) #TODO
        
    def reset(self, seed = None):
        
        pass
    
    def step(self, action):
        
        pass
    
    def _compute_reward(self, last_state, current_state):
        
        pass
        
    def render(self):
        
        pass