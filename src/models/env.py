import gymnasium as gym
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from chiller_model import ChillerTimeSeriesModel
from electrical_model import ElectricalTimeSeriesModel
from boiler_model import BoilerTimeSeriesModel
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback




class SurrogateEnv(gym.Env):
    
    def __init__(
        self, 
        model_folder = '../pickled_models', 
        demand_data:pd.DataFrame = None,
        state_csv_columns = ['heatingGroupFmu.TAmb', 'heatingGroupFmu.mSteDem', 'heatingGroupFmu.boilerFWPum.port_a.p', 
                             'chillersFMU.TWetBul', 'chillersFMU.mChiWat', 'chillersFMU.Qchp', 'electricalGroup.Ir', 'electricalGroup.Pdem',
                              'electricalGroup.Pgen', 'electricalGroup.Ppla', 'electricalGroup.Soc','CO2_ele','CO2_gas', 'Ele_buy', 'Ele_sell', 'Gas_buy' ,'Dem_cha'
         ], # 17 spaces
        price_columns = ['CO2_ele','CO2_gas', 'Ele_buy', 'Ele_sell', 'Gas_buy' ,'Dem_cha']
    ):
        
        self.model_folder = model_folder
        self.demand_data = demand_data
        self.state_csv_columns = state_csv_columns
        self.price_columns = price_columns
        
        self.observation_space = gym.spaces.Box(low = 0, high = 1, shape = (17, )) #TODO
        self.action_space = gym.spaces.Box(low = 0, high = 1, shape = (5, )) #TODO
        
        self.max_timesteps_ = 4000 #should this be 720 hours /month? or 8760 hrs/year?
        
        self.setup_models()
        
        # For plotting rewards
        self.rewards_log = {
            'reward_total': [],
            'grid_reward': [],
            'fuel_rewards': [],
            'peakDem_rewards': [],
            'co2_reward': [],
            'safe_reward': []
        }

        # Adding boiler, chiller, and electrical outputs as instance attributes for access
        self.boiler_output = None
        self.chiller_output = None
        self.electrical_output = None

        
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
            self.electrical_scaler_X.transform(np.concatenate([state.to_numpy()[6:11], np.zeros(3)]).reshape(1, -1))[0][:-3],
            state[11:]

            
        ])



        
        # Resetting self.state
        #self.state = state
        self.state.loc[self.state_csv_columns] = state
        #print("state_init:", self.state)




        self.max_p_grid = [0]
        return state, info
    
    def step(self, action):
        
        #y_chp, ext_on, abs_on, abschi_on, y_val, bat_chg = action[0], action[1], action[2], action[3], action[4], action[5]
        y_chp, ext_on, abschi_on, y_val, bat_chg = action[0], action[1], action[2], action[3], action[4]
        #print("actions:",  action[0], action[1], action[2], action[3], action[4])
        
        # Boiler
        
        boiler_state = np.array([self.state['heatingGroupFmu.TAmb'], self.state['heatingGroupFmu.mSteDem'], self.state['heatingGroupFmu.boilerFWPum.port_a.p']])
        #abs chiOn = absOn , both are same variables that are action for both boiler and chiller
        boiler_action = self.boiler_scaler_X.transform(np.concatenate([self.state.to_numpy()[:3], np.array([y_chp, int(ext_on > 0.5), int(abschi_on > 0.5)])]).reshape(1, -1))[0][-3:]   
        self.boiler_output = self.boiler_scaler_y.inverse_transform(self.boiler_model.step(boiler_state, boiler_action).reshape(1, -1))[0]
        
        # Chiller
        chiller_state = np.array([self.state['chillersFMU.TWetBul'], self.state['chillersFMU.mChiWat'], self.state['chillersFMU.Qchp']])
        chiller_action = self.chiller_scaler_X.transform(np.concatenate([self.state.to_numpy()[3:6], np.array([int(abschi_on > 0.5), y_val])]).reshape(1, -1))[0][-2:]
        self.chiller_output = self.chiller_scaler_y.inverse_transform(self.chiller_model.step(chiller_state, chiller_action).reshape(1, -1))[0]
        
        # Electrical
        electrical_state = np.array([self.state['electricalGroup.Ir'], self.state['electricalGroup.Pdem'], self.state['electricalGroup.Ppla'], self.state['electricalGroup.Pgen'], self.state['electricalGroup.Soc']])
        battery_ch = int(bat_chg < 0.33)
        battery_stb = int(bat_chg >= 0.33 and bat_chg < 0.66)
        battery_dist = int(bat_chg > 0.66)
        electrical_action = self.electrical_scaler_X.transform(np.concatenate([self.state.to_numpy()[6:11], np.array([battery_ch, battery_stb, battery_dist])]).reshape(1, -1))[0][-3:]
        self.electrical_output =  self.electrical_scaler_y.inverse_transform(self.electrical_model.step(electrical_state, electrical_action).reshape(1, -1))[0]
        
        # Update state
        self.state = self.demand_data.iloc[self.idx + self.timesteps + 1][self.state_csv_columns]
        
        # Create a copy of self.state before modifying it
        updated_state = self.state.copy()
        #print("state_preUpdate:", updated_state)


        updated_state.loc['chillersFMU.Qchp'] = self.boiler_output[6]
        updated_state.loc['electricalGroup.Pgen'] = self.boiler_output[0] + self.boiler_output[1]
        updated_state.loc['electricalGroup.Ppla'] = -(sum(self.chiller_output[:4]) + self.chiller_output[6])
        updated_state.loc['electricalGroup.Soc'] = self.electrical_output[1]
        
        #state = self.state
        #print("state_postUpdate_values:",  boiler_output[6], (boiler_output[0] + boiler_output[1]),  -(sum(chiller_output[:4]) + chiller_output[6]), electrical_output[1] )

       # Transform updated state
        transformed_state = np.concatenate([
            self.boiler_scaler_X.transform(np.concatenate([updated_state.to_numpy()[:3], np.zeros(3)]).reshape(1, -1))[0][:-3],
            self.chiller_scaler_X.transform(np.concatenate([updated_state.to_numpy()[3:6], np.zeros(2)]).reshape(1, -1))[0][:-2],
            self.electrical_scaler_X.transform(np.concatenate([updated_state.to_numpy()[6:11], np.zeros(3)]).reshape(1, -1))[0][:-3],
            updated_state[11:] # Adding price columns directly to the state
        ])
        #print("state_postUpdate:", transformed_state)
        # Assign transformed state back to a new DataFrame instead of overwriting original state directly
        # Use self.state as a new DataFrame to maintain consistency
        #self.state = pd.Series(transformed_state, index=self.state_csv_columns)
        # Assign transformed state back to self.state
        self.state.loc[self.state_csv_columns] = transformed_state

        #print("state_postUpdate:", transformed_state)
        #print("boiler out:", boiler_output, "chiller output:", chiller_output,"electrical output:", electrical_output)


        # Check for NaN values and replace them with zeros to avoid invalid operations
        if np.isnan(transformed_state).any():
            state = np.nan_to_num(transformed_state)

        # Compute reward
        self.boiler_output = pd.DataFrame(self.boiler_output.reshape(1, -1), columns = ['heatingGroupFmu.pSteTur', 'heatingGroupFmu.pGasTur', 'heatingGroupFmu.mSteBoi', 
           'heatingGroupFmu.mSteChp', 'heatingGroupFmu.mFueChp', 'heatingGroupFmu.mFueBoi', 'heatingGroupFmu.qAbsChi', 
           'heatingGroupFmu.preChp.p','heatingGroupFmu.preBoi.p', 'heatingGroupFmu.chp.botCycExp.QsteAbs',
          'heatingGroupFmu.chp.botCycExp.Qlos','heatingGroupFmu.chp.botCycExp.exhQ.y','heatingGroupFmu.qFueChp'] )
        self.chiller_output = pd.DataFrame(self.chiller_output.reshape(1, -1), columns = ['chillersFMU.pPumAbs', 'chillersFMU.pFanAbs', 'chillersFMU.pPum', 'chillersFMU.pFan', 
            'chillersFMU.TchiAbsSup', 'chillersFMU.TchiSup', 'chillersFMU.pChi', 
           'chillersFMU.QchiAbs','chillersFMU.mAbsChi','chillersFMU.mCenChi' ]  )
        self.electrical_output = pd.DataFrame(self.electrical_output.reshape(1, -1), columns = ['electricalGroup.Ppv', 'electricalGroup.Soc_duplicate', 'electricalGroup.Pbat', 'electricalGroup.Pgri.real'])
        
        self.max_p_grid = self.max_p_grid + [self.electrical_output['electricalGroup.Pgri.real'][0]]
        if len(self.max_p_grid) > 720:
            self.max_p_grid.pop(0)
        
        # Compute reward and log results
        reward, ind_rews = self._compute_reward(self.boiler_output, self.chiller_output, self.electrical_output, self.demand_data) ##revised as self.dem...
        self.timesteps+=1
        
        # Log reward parameters for plotting
        self.rewards_log['reward_total'].append(reward)
        self.rewards_log['grid_reward'].append(ind_rews[0])
        self.rewards_log['fuel_rewards'].append(ind_rews[1])
        self.rewards_log['peakDem_rewards'].append(ind_rews[2])
        self.rewards_log['co2_reward'].append(ind_rews[3])
        self.rewards_log['safe_reward'].append(ind_rews[4])
        
        done = self.timesteps >= self.max_timesteps_
        trunc = False
        #info = dict(grid_reward = ind_rews[0], fuel_rewards = ind_rews[1], peakDem_rewards = ind_rews[2], co2_reward = ind_rews[3], safe_reward = ind_rews[4])
        
         # Include boiler, chiller, and electrical outputs in the info dictionary
        info = {
        'grid_reward': ind_rews[0], 
        'fuel_rewards': ind_rews[1], 
        'peakDem_rewards': ind_rews[2], 
        'co2_reward': ind_rews[3], 
        'safe_reward': ind_rews[4],
        'boiler_output': self.boiler_output, 
        'chiller_output': self.chiller_output, 
        'electrical_output': self.electrical_output,
        'state_updated': self.state
        }
        

        return self.state.to_numpy(), reward, done, trunc, info
        
    
    def _compute_reward(self, boiler_output, chiller_output, electrical_output, demand_data):
        # Weight array 
        self.s = [1, 0.5, 0.25, 0.5, 0.5, 0.5] # ratios/factors
        #[grid_reward, fuel_reward, peak_demand, emissions, system stability p , t ]

        # energy cost 
        buy_price = demand_data['Ele_buy'][self.idx + self.timesteps] # $/kwh
        sell_price = demand_data['Ele_sell'][self.idx + self.timesteps] # $/kwh
        dem_price = demand_data['Dem_cha'][self.idx + self.timesteps] # $/kwh
        gas_price = demand_data['Gas_buy'][self.idx + self.timesteps] # $/kg

        #print("PriceSignals:",  buy_price, sell_price, dem_price, gas_price )

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
        grid_reward = self.s[0] * ((1-(Cgrid/Cgrid_max))**2)
       

        #gas 
        Cgas = m_gas * gas_price
        Cgas_max = 6400 * gas_price # max gas consumption 6400 kg/hr
        fuel_rewards = self.s[1] * min((Cgas_max/Cgas), 20) if Cgas != 0 else 0
        

        #demand (monthly peak demand)
        Cdem = max(self.max_p_grid)/1000 * dem_price if max(self.max_p_grid) != 0 else 1e-5
        CdemMax = 21*10**3 * dem_price
        peakDem_rewards = self.s[2] * min((CdemMax/Cdem), 20)
        

        #Carbon emission reward
        emissions = ((p_grid * CO2_ele) if p_grid>0 else 0) + (m_gas * CO2_gas)
        emissions_max =21*(10**3) * CO2_ele +  6400 * CO2_gas
        co2_reward = self.s[3] * (emissions_max/emissions) if emissions != 0 else 0
        

        #system stability
        safe_reward = self.s[4] * p_norm**2  + self.s[5] * t_norm**2
        

        reward_total = (grid_reward + fuel_rewards + peakDem_rewards + co2_reward + safe_reward)/6
        # reward_total = grid_reward + fuel_rewards + co2_reward + safe_reward
        
        
        # print("Outputs:",  p_grid, m_gas, p_norm, t_norm )
        # print("Grid rewards:", Cgrid, Cgrid_max, grid_reward)
        # print("fuel rewards:", Cgas, Cgas_max, fuel_rewards)
        # print("Demand rewards:", Cdem , CdemMax , peakDem_rewards)
        # print("co2 rewards:", emissions , emissions_max ,co2_reward)
        # print("Stability:", p_norm,  t_norm, safe_reward)
        # print("total reward:",reward_total)

        return reward_total, [grid_reward, fuel_rewards, peakDem_rewards, co2_reward, safe_reward]
        
    def render(self):
        pass

   
    

if __name__ == "__main__":
        
    # Load the data
    electrical_data = pd.read_csv('../../Electrical_annual.csv')

    electrical_data.drop_duplicates(subset = ['Time'], inplace = True)
    electrical_data.reset_index(drop = True, inplace = True)
    chiller_data = pd.read_csv('../../cooling_annual.csv').iloc[:electrical_data.shape[0]]
    chiller_data.index = electrical_data.index
    boiler_data = pd.read_csv('../../Heating_annual.csv').iloc[:electrical_data.shape[0]]
    boiler_data.index = electrical_data.index

    #price and emissions data
    price_data = pd.read_csv('../../EmissionsPrice.csv').iloc[:electrical_data.shape[0]]
    price_data.index = electrical_data.index

    
    demand_data = pd.DataFrame(index = electrical_data.index)
    demand_data = demand_data.join(boiler_data[['heatingGroupFmu.TAmb', 'heatingGroupFmu.mSteDem', 'heatingGroupFmu.boilerFWPum.port_a.p']])
    demand_data = demand_data.join(chiller_data[['chillersFMU.TWetBul', 'chillersFMU.mChiWat', 'chillersFMU.Qchp',]])
    demand_data = demand_data.join(electrical_data[[ 'electricalGroup.Ir','electricalGroup.Pdem', 'electricalGroup.Pgen', 'electricalGroup.Ppla', 'electricalGroup.Soc']])
    demand_data = demand_data.join(price_data[[ 'CO2_ele','CO2_gas', 'Ele_buy', 'Ele_sell', 'Gas_buy' ,'Dem_cha']])
    #print(demand_data)

    # env = SurrogateEnv(demand_data = demand_data)
    # model = PPO('MlpPolicy', env, verbose = 1)
    # model.learn(50000)
    

    #Step 1: Set up the environment and the model
    # Wrap the environment to monitor training progress
    env = SurrogateEnv(demand_data=demand_data)
    env = Monitor(env)

    # Initialize PPO model with TensorBoard logging
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    
    # Set up evaluation callback to save the best model
    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

    # Step 2: Train the model
    model.learn(total_timesteps=100000, callback=eval_callback)

    #If you set total_timesteps=100000, this means approximately 100000 / 8760 ≈ 11.4 full episodes (i.e., about 11 full years of training).

    

    # Step 3: Save the trained model
    model.save("ppo_trained_agent2")
    print("Model saved as 'ppo_trained_agent2.zip'")
