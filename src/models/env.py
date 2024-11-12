import gymnasium as gym
import pandas as pd
import numpy as np
import pickle
from chiller_model import ChillerTimeSeriesModel
from electrical_model import ElectricalTimeSeriesModel
from boiler_model import BoilerTimeSeriesModel
from stable_baselines3 import PPO

class SurrogateEnv(gym.Env):
    
    def __init__(
        self, 
        model_folder = '../pickled_models', 
        demand_data:pd.DataFrame = None,
        state_csv_columns = ['heatingGroupFmu.TAmb', 'heatingGroupFmu.mSteDem', 'heatingGroupFmu.boilerFWPum.port_a.p', 
                             'chillersFMU.TWetBul', 'chillersFMU.mChiWat', 'chillersFMU.Qchp', 'electricalGroup.Ir', 'electricalGroup.Pdem',
                              'electricalGroup.Pgen', 'electricalGroup.Ppla', 'electricalGroup.Soc', 
         ], # 17 spaces
        price_columns = ['CO2_ele','CO2_gas', 'Ele_buy', 'Ele_sell', 'Gas_buy' ,'Dem_cha']
    ):
        
        self.model_folder = model_folder
        self.demand_data = demand_data
        self.state_csv_columns = state_csv_columns
        self.price_columns = price_columns
        
        self.observation_space = gym.spaces.Box(low = 0, high = 1, shape = (11, )) #TODO
        self.action_space = gym.spaces.Box(low = 0, high = 1, shape = (6, )) #TODO
        
        self.max_timesteps_ = 2160 #should this be 720 hours /month? or 8760 hrs/year?
        
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
        self.max_p_grid = [0]
        
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
        self.max_p_grid = [0]
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
        self.state.loc['electricalGroup.Pgen'] = boiler_output[0] + boiler_output[1]
        self.state.loc['electricalGroup.Ppla'] = -(sum(chiller_output[:4]) + chiller_output[6])
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
        
        self.max_p_grid = self.max_p_grid + [electrical_output['electricalGroup.Pgri.real'][0]]
        if len(self.max_p_grid) > 720:
            self.max_p_grid.pop(0)
        reward = self._compute_reward(boiler_output, chiller_output, electrical_output, demand_data)
        self.timesteps+=1
        
        done = self.timesteps >= self.max_timesteps_
        trunc = done
        info = dict(electrical_output = electrical_output, boiler_output = boiler_output, chiller_output = chiller_output)
        
        return self.state.to_numpy(), reward, done, trunc, info
        
        
    
    def _compute_reward(self, boiler_output, chiller_output, electrical_output, demand_data):
        # Weight array 
        self.s = [1, 0.5, 0, 0, 1, 0.2] # ratios/factors
        #[grid_reward, fuel_reward, peak_demand, emissions, system stability p , t ]


        # energy cost 
        buy_price = demand_data['Ele_buy'][self.idx + self.timesteps] # $/kwh
        sell_price = demand_data['Ele_sell'][self.idx + self.timesteps] # $/kwh
        dem_price = demand_data['Dem_cha'][self.idx + self.timesteps] # $/kwh
        gas_price = demand_data['Gas_buy'][self.idx + self.timesteps] # $/kg

        # Carbon emissions
        CO2_ele = demand_data['CO2_ele'][self.idx + self.timesteps]*1000 # kg co2/kW
        CO2_gas = demand_data['CO2_gas'][self.idx + self.timesteps]*1000 # kg co2/kg fuel

       
        # outputs:
        p_grid = electrical_output['electricalGroup.Pgri.real'][0]/1000 # in kWh
        m_gas = (boiler_output['heatingGroupFmu.mFueChp'][0]+ boiler_output['heatingGroupFmu.mFueBoi'][0])*3600 #in kg/h
        p_norm = (boiler_output['heatingGroupFmu.preChp.p'][0] + boiler_output['heatingGroupFmu.preBoi.p'][0])/(2*9*10**5) # pa/pa pressure setpoint 
        t_norm = (chiller_output['chillersFMU.TchiAbsSup'][0] + chiller_output['chillersFMU.TchiSup'][0])/(2*280.15) # K/K temp setpoint


        #energy cost reward = electricity + gas+ demand costs
        #grid 
        Cgrid =  ((p_grid * buy_price) if p_grid>0 else (p_grid * sell_price))
        Cgrid_max = 21*10**3 * buy_price  # maximum campus demand 21MW; CHP capacity = 15MW + 2MW of PV
        grid_reward = self.s[1] * ((1-(Cgrid/Cgrid_max))**2)

        #gas 
        Cgas = m_gas * gas_price
        Cgas_max = 6400 * gas_price # max gas consumption 6400 kg/hr
        fuel_rewards = self.s[2] * (Cgas_max/Cgas)

        #demand (monthly peak demand)
        Cdem =  max(self.max_p_grid) * dem_price
        CdemMax = 21*10**3 * dem_price
        peakDem_rewards = self.s[3] * (CdemMax/Cdem)


        #Carbon emission reward
        emissions = ((p_grid * CO2_ele) if p_grid>0 else 0) + (m_gas * CO2_gas)
        emissions_max = 6400 * CO2_gas
        co2_reward = self.s[4] * (emissions_max/emissions)


        #system stability
        safe_reward = self.s[4] * p_norm**2  + self.s[5] * t_norm**2

        reward_total = grid_reward + fuel_rewards + peakDem_rewards + co2_reward + safe_reward
        # reward_total = grid_reward + fuel_rewards + co2_reward + safe_reward

        
        
        return reward_total
        
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

    #price and emissions data
    price_data = pd.read_csv('../../EmissionsPrice.csv').iloc[:electrical_data.shape[0]]
    price_data.index = electrical_data.index

    
    demand_data = pd.DataFrame(index = electrical_data.index)
    demand_data = demand_data.join(boiler_data[['heatingGroupFmu.TAmb', 'heatingGroupFmu.mSteDem', 'heatingGroupFmu.boilerFWPum.port_a.p']])
    demand_data = demand_data.join(chiller_data[['chillersFMU.TWetBul', 'chillersFMU.mChiWat', 'chillersFMU.Qchp',]])
    demand_data = demand_data.join(electrical_data[[ 'electricalGroup.Ir','electricalGroup.Pdem', 'electricalGroup.Pgen', 'electricalGroup.Ppla', 'electricalGroup.Soc']])
    demand_data = demand_data.join(price_data[[ 'CO2_ele','CO2_gas', 'Ele_buy', 'Ele_sell', 'Gas_buy' ,'Dem_cha']])

    env = SurrogateEnv(demand_data = demand_data)
    PPO('MlpPolicy', env, verbose = 1).learn(10000)