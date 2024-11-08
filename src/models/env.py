import gymnasium as gym
import pandas as pd
import numpy as np
import pickle
from chiller_model import ChillerTimeSeriesModel
from electrical_model import ElectricalTimeSeriesModel
from boiler_model import BoilerTimeSeriesModel

class SurrogateEnv(gym.Env):
    
    def __init__(
        self, 
        model_folder = '../pickled_models', 
        demand_data:pd.DataFrame = None,
        state_csv_columns = ['heatingGroupFmu.TAmb', 'heatingGroupFmu.mSteDem', 'heatingGroupFmu.boilerFWPum.port_a.p', 'chillersFMU.TWetBul', 'chillersFMU.mChiWat', 'chillersFMU.Qchp', 'electricalGroup.Ir', 'electricalGroup.Pdem', 'electricalGroup.Pgen', 'electricalGroup.Ppla', 'electricalGroup.Soc']
    ):
        
        self.model_folder = model_folder
        self.demand_data = demand_data
        self.state_csv_columns = state_csv_columns
        
        self.observation_space = gym.spaces.Box(low = 0, high = 1, shape = (10, )) #TODO
        self.action_space = gym.spaces.Box(low = 0, high = 1, shape = (6, )) #TODO
        
        self.max_timesteps_ = 2160
        
        self.setup_models()
        
    def setup_models(self):
        
        self.boiler_model = BoilerTimeSeriesModel.load_from_checkpoint(checkpoint_path=self.model_folder + '/boiler_model.ckpt')
        self.boiler_scaler_X = pickle.load(open(self.model_folder + '/boiler_scaler_X.pkl', 'rb'))
        self.boiler_scaler_y = pickle.load(open(self.model_folder + '/boiler_scaler_y.pkl', 'rb'))
        self.chiller_model = ChillerTimeSeriesModel.load_from_checkpoint(checkpoint_path=self.model_folder + '/chiller_model.ckpt')
        self.chiller_scaler_X = pickle.load(open(self.model_folder + '/chiller_scaler_X.pkl', 'rb'))
        self.chiller_scaler_y = pickle.load(open(self.model_folder + '/chiller_scaler_y.pkl', 'rb'))
        self.electrical_model = ElectricalTimeSeriesModel.load_from_checkpoint(checkpoint_path=self.model_folder + '/electrical_model.ckpt')
        self.electrical_scaler_X = pickle.load(open(self.model_folder + '/electrical_scaler_X.pkl', 'rb'))
        self.electrical_scaler_y = pickle.load(open(self.model_folder + '/electrical_scaler_y.pkl', 'rb'))
        
    def reset(self, seed = None):
        
        '''
        For Boiler: Need Tamb, mSteDem, preFw
        For Chiller: Need Twet, mChiWat, Qchp
        For Electrical: Need Ir, pdem, pgen, ppla, SOC
        '''
        
        self.idx = np.random.randint(0, len(self.demand_data) - self.max_timesteps_)
        self.timesteps = 0
        state = self.demand_data.iloc[self.idx][self.state_csv_columns]
        info = {}
        
        self.state = state
        state = np.concatenate([
            self.boiler_scaler_X.transform(np.concatenate([state.to_numpy()[:3], np.zeros(3)]).reshape(1, -1))[0][:-3],
            self.chiller_scaler_X.transform(np.concatenate([state.to_numpy()[3:6], np.zeros(2)]).reshape(1, -1))[0][:-2],
            self.electrical_scaler_X.transform(np.concatenate([state.to_numpy()[6:], np.zeros(3)]).reshape(1, -1))[0][:-3]
            
        ])
        
        self.state.loc[self.state_csv_columns] = state
        return state, info
    
    def step(self, action):
        
        y_chp, ext_on, abs_on, abschi_on, y_val, bat_chg = action[0], action[1], action[2], action[3], action[4], action[5]
        
        # Boiler
        
        boiler_state = np.array([self.state['heatingGroupFmu.TAmb'], self.state['heatingGroupFmu.mSteDem'], self.state['heatingGroupFmu.boilerFWPum.port_a.p']])
        boiler_action = self.boiler_scaler_X.transform(np.concatenate([self.state.to_numpy()[:3], np.array([y_chp, int(ext_on > 0.5), int(ext_on > 0.5)])]).reshape(1, -1))[0][-3:]
           
        boiler_output = self.boiler_scaler_y.inverse_transform(self.boiler_model.step(boiler_state, boiler_action).reshape(1, -1))[0]
        
        # Chiller
        chiller_state = np.array([self.state['chillersFMU.TWetBul'], self.state['chillersFMU.mChiWat'], self.state['chillersFMU.Qchp']])
        chiller_action = self.chiller_scaler_X.transform(np.concatenate([self.state.to_numpy()[3:6], np.array([int(abschi_on > 0.5), y_val])]).reshape(1, -1))[0][-2:]
        chiller_output = self.chiller_scaler_y.inverse_transform(self.chiller_model.step(chiller_state, chiller_action).reshape(1, -1))[0]
        
        # Electrical
        electrical_state = np.array([self.state['electricalGroup.Ir'], self.state['electricalGroup.Pdem'], self.state['electricalGroup.Pgen'], self.state['electricalGroup.Ppla'], self.state['electricalGroup.Soc']])
        battery_ch = int(bat_chg < 0.33)
        battery_stb = int(bat_chg >= 0.33 and bat_chg < 0.66)
        battery_dist = int(bat_chg > 0.66)
        electrical_action = self.electrical_scaler_X.transform(np.concatenate([self.state.to_numpy()[6:], np.array([battery_ch, battery_stb, battery_dist])]).reshape(1, -1))[0][-3:]
        electrical_output =  self.electrical_scaler_y.inverse_transform(self.electrical_model.step(electrical_state, electrical_action).reshape(1, -1))[0]
        
        # Update state
        self.state = self.demand_data.iloc[self.idx + self.timesteps + 1][self.state_csv_columns]
        self.state.loc['chillersFMU.Qchp'] = boiler_output[6]
        self.state.loc['electricalGroup.Pgen'] = sum(chiller_output[:4]) + chiller_output[6]
        self.state.loc['electricalGroup.Ppla'] = boiler_output[0] + boiler_output[1]
        self.state.loc['electricalGroup.Soc'] = electrical_output[1]
        state = self.state
        state = np.concatenate([
            self.boiler_scaler_X.transform(np.concatenate([state.to_numpy()[:3], np.zeros(3)]).reshape(1, -1))[0][:-3],
            self.chiller_scaler_X.transform(np.concatenate([state.to_numpy()[3:6], np.zeros(2)]).reshape(1, -1))[0][:-2],
            self.electrical_scaler_X.transform(np.concatenate([state.to_numpy()[6:], np.zeros(3)]).reshape(1, -1))[0][:-3]
            
        ])
        
        self.state.loc[self.state_csv_columns] = state
        # Compute reward
        boiler_output = pd.DataFrame(boiler_output.reshape(1, -1), columns = ['heatingGroupFmu.pSteTur', 'heatingGroupFmu.pGasTur', 'heatingGroupFmu.mSteBoi', 
           'heatingGroupFmu.mSteChp', 'heatingGroupFmu.mFueChp', 'heatingGroupFmu.mFueBoi', 'heatingGroupFmu.qAbsChi', 
           'heatingGroupFmu.preChp.p','heatingGroupFmu.preBoi.p', 'heatingGroupFmu.chp.botCycExp.QsteAbs',
          'heatingGroupFmu.chp.botCycExp.Qlos','heatingGroupFmu.chp.botCycExp.exhQ.y','heatingGroupFmu.qFueChp'] )
        chiller_output = pd.DataFrame(chiller_output.reshape(1, -1), columns = ['chillersFMU.pPumAbs', 'chillersFMU.pFanAbs', 'chillersFMU.pPum', 'chillersFMU.pFan', 
            'chillersFMU.TchiAbsSup', 'chillersFMU.TchiSup', 'chillersFMU.pChi', 
           'chillersFMU.QchiAbs','chillersFMU.mAbsChi','chillersFMU.mCenChi' ]  )
        electrical_output = pd.DataFrame(electrical_output.reshape(1, -1), columns = ['electricalGroup.Ppv', 'electricalGroup.Soc_duplicate', 'electricalGroup.Pbat', 'electricalGroup.Pgri.real'])
        
        reward = self._compute_reward(boiler_output, chiller_output, electrical_output)
        self.timesteps+=1
        
        done = self.timesteps >= self.max_timesteps_
        trunc = done
        info = dict(electrical_output = electrical_output, boiler_output = boiler_output, chiller_output = chiller_output)
        
        return self.state.to_numpy(), reward, done, trunc, info
        
        
    
    def _compute_reward(self, boiler_output, chiller_output, electrical_output):
        
        return 0 #TODO
        
    def render(self):
        
        pass
    

if __name__ == "__main__":
        
            
    # Load the data
    electrical_data = pd.read_csv('../../ElectricalGroupResults.csv')

    electrical_data.drop_duplicates(subset = ['Time'], inplace = True)
    electrical_data.reset_index(drop = True, inplace = True)
    chiller_data = pd.read_csv('../../chillerGroupResults.csv').iloc[:electrical_data.shape[0]]
    chiller_data.index = electrical_data.index
    boiler_data = pd.read_csv('../../HeatingGroupResults.csv').iloc[:electrical_data.shape[0]]
    boiler_data.index = electrical_data.index
    
    demand_data = pd.DataFrame(index = electrical_data.index)
    demand_data = demand_data.join(boiler_data[['heatingGroupFmu.TAmb', 'heatingGroupFmu.mSteDem', 'heatingGroupFmu.boilerFWPum.port_a.p']])
    demand_data = demand_data.join(chiller_data[['chillersFMU.TWetBul', 'chillersFMU.mChiWat', 'chillersFMU.Qchp',]])
    demand_data = demand_data.join(electrical_data[[ 'electricalGroup.Ir','electricalGroup.Pdem', 'electricalGroup.Pgen', 'electricalGroup.Ppla', 'electricalGroup.Soc']])
        

    env = SurrogateEnv(demand_data = demand_data)
    state, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, trunc, info = env.step(action)
        print(reward)
        print(state)
        print(info)
        print(done)
        print(trunc)
        print('-------------------')