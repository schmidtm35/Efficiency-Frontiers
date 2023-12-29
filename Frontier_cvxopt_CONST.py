# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import solvers, blas
solvers.options['show_progress'] = False

year = [2017]

teams= [1610612738, 1610612744, 1610612766, 1610612751, 1610612748,\
        1610612755, 1610612763, 1610612737, 1610612743, 1610612750,\
        1610612756, 1610612745, 1610612741, 1610612752, 1610612746,\
        1610612754, 1610612765, 1610612739, 1610612742, 1610612747,\
        1610612760, 1610612759, 1610612740, 1610612753, 1610612758,\
        1610612761, 1610612762, 1610612764, 1610612749, 1610612757]

team_return_df = []
team_volatility_df = []
team_Sharpe_df = []
team_volatility_df = []
team_Sharpe_df = []
team_return_diff = []
team_return_df = []
game = []
team_risk_diff = []  
  
for y in year:
    
    for k in teams:
        data = pd.read_csv('/Users/martyschmidt/Box/Basketball/Tanking/team_data/team_plus_minus_'+ str(k) + '_' + str(y) + '.csv' )
        data.columns = data.columns.str.lower()
        data.drop(data.columns[0], inplace=True, axis = 1)
        data.rename(columns={'player_id_time': 'min'}, inplace=True)
    
    
        #create a dictionary of teams - seperate DataFrames
        
        events = {j: u for j, u in data.groupby('game_id')}
        
        
        for key in events:
            
            clean = events[key]
            player_weights = clean.groupby('player_id')['min'].max().div(clean.groupby('player_id')['time'].max()).div(5)
            player_weights=player_weights.reset_index()
            clean.drop(['team_id', 'min', 'game_id', 'year'], inplace=True, axis=1)
            table = clean.pivot_table(index='time', columns='player_id')
        
        #drop any column where the NaNs are more than 80%
        
        #fill existing Nan with column's mean value
            table = table.loc[:, table.isnull().sum() < 0.8*table.shape[0]]
            table.fillna(table.mean(), inplace=True)
        # calculate daily and annual returns of the stocks
            returns_daily = table
            returns_annual = returns_daily.mean() * 48
        
            returns_annual.index = returns_annual.index.droplevel(0)
            returns_annual.to_frame()
            returns_annual.rename(columns={0:'Return'}, inplace=True)
            
        # get daily and annual covariance of returns of the stock
            cov_daily = returns_daily.cov()
            cov_annual = cov_daily
            cov_annual.columns = cov_annual.columns.droplevel(0)
            cov_annual.index = cov_annual.index.droplevel(0)
            
        #calculate actual return
            actual_return_half = returns_annual.reset_index()
            actual_return_half.rename(columns={0: 'return'}, inplace=True)
            actual_return = pd.merge(actual_return_half,  player_weights, how='left', on='player_id')
            actual_return.rename(columns={0: 'min'}, inplace=True)
            actual_return['expected_return_id'] = actual_return['return'] * actual_return['min']
            Return = np.dot(actual_return['min'], returns_annual)
            team_return_df.append(Return)
        
            Volatility = np.sqrt(np.dot(actual_return['min'].T, np.dot(cov_daily, actual_return['min'])))
            team_volatility_df.append(Volatility)
            
            Sharpe = Return / Volatility
            team_Sharpe_df.append(Sharpe)
            
        # use the min, max values to locate and create the two special portfolios
            actual_portfolio = (Volatility, Return)
            actual_portfolio = pd.DataFrame({'Volatility': [Volatility], 'Returns': [Return]},
                                  index=[1])
            players = {k: v for k, v in returns_annual.groupby('player_id')}
            
            def random_wieghts(n):
            
                a = np.random.uniform(0, 0.20, size=len(players))
                return a/a.sum()
            def initial_portfolio(returns_annual):
                #monthly_returns = data.resample('BM', how=lambda x: (x[-1]/x[0])-1)
                
                cov = np.matrix(cov_annual)
                expected_returns = np.matrix(returns_annual)
                wieghs = np.matrix(random_wieghts(expected_returns.shape[1]))
                
                mu = wieghs.dot(expected_returns.T)
                sigma = np.sqrt(wieghs * cov.dot(wieghs.T))
                
                return mu[0,0],sigma[0,0]
            n_portfolios = 1000
            means, stds = np.column_stack([initial_portfolio(returns_annual) for _ in range(n_portfolios)])
            plt.figure(figsize=(8, 8), dpi=80)
            sharp=means/stds
        
            
            def optimal_portfolio(returns_annual):  
                n = len(players)  
                N = 100
                mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  
                # Convert to cvxopt matrices  
                S = opt.matrix(cov_annual.values) 
                pbar = opt.matrix(returns_annual.values)  
                # Create constraint matrices  
                
                Identity = opt.matrix(0.0, (n,n)) 
                Identity[::n+1] = 1.0 
                G = opt.matrix([-Identity, Identity])
                lower = opt.matrix(0.0, (n,1)) 
                upper = opt.matrix(0.20, (n,1))
                h = opt.matrix([-lower, upper])
                A = opt.matrix(1.0, (1, n))  
                b = opt.matrix(1.0) 
        
                # Calculate efficient frontier weights using quadratic programming  
                portfolios = [solvers.qp(mu*S, -pbar, G,  h, A, b)['x']  
                              for mu in mus]  
                ## CALCULATE RISKS AND RETURNS FOR FRONTIER  
                returns = [blas.dot(pbar, x) for x in portfolios]  
                risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]  
                ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE  
                m1 = np.polyfit(returns, risks, 2)  
                x1 = np.sqrt(m1[2] / m1[0])  
        
                # CALCULATE THE OPTIMAL PORTFOLIO  
                wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  
                return np.asarray(wt), returns, risks
        
            weights, returns, risks = optimal_portfolio(returns_annual)
            
            plt.plot(stds, means, 'o')  
            plt.plot(risks, returns, 'y--')
            plt.plot(actual_portfolio['Volatility'], actual_portfolio['Returns'], 'r-o')
            plt.ylabel('mean')  
            plt.xlabel('std')
            
            eff_risk = pd.DataFrame(risks)
            eff_risk.rename(columns={0:'Eff_Risk'}, inplace=True)
            eff_return = pd.DataFrame(returns)  
            eff_return.rename(columns={0:'Eff_Return'}, inplace=True) 
            eff_front = eff_return.join(eff_risk)
        
            #locate closest efficient portfolio to the actual portfolio
            risk_diff = eff_front.loc[(eff_front['Eff_Risk']-Volatility).abs().argsort()[:1]]
            return_diff = eff_front.loc[(eff_front['Eff_Return']-Return).abs().argsort()[:1]]
            risk_diff['act_risk'] = Volatility
            risk_diff['team'] = k
            risk_diff['game'] = key
            risk_diff['season'] =  y
            return_diff['act_return'] = Return
            return_diff['team'] = k
            return_diff['game'] = key
            return_diff['season'] = y
            
            #merge all games
            team_risk_diff.append(risk_diff)
            team_return_diff.append(return_diff)
            
df_risk = pd.concat(team_risk_diff)
df_return = pd.concat(team_return_diff)

df_risk['team'].value_counts()

df_risk.drop(['Eff_Return'], inplace=True, axis=1)
df_return.drop(['Eff_Risk'], inplace=True, axis=1)

df_risk['risk_diff'] = df_risk['Eff_Risk'] - df_risk['act_risk']
df_return['return_diff'] = df_return['Eff_Return'] - df_return['act_return']

df_risk_return = pd.merge(df_risk, df_return, on=['team', 'game', 'season'])

save = ('/Users/martyschmidt/Box/Basketball/Tanking/Game_Portfolios/team_risk_return_more2017.xls' )

df_risk_return.to_excel(save)   

