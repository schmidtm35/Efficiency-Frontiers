# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

game_df = pd.read_csv("/Users/martinbschmidt/Box/Basketball/data/court_2018_2019_pbp.csv")
game_df.columns = game_df.columns.str.lower()
game_df.head()
game_df.drop(['eventmsgactiontype','eventmsgtype', 'eventnum'], axis=1, inplace=True)
game_df.replace(np.nan, 0, inplace=True)

game_df['home_diff'] = game_df.home_score - game_df.away_score
game_df.sort_values(by=['game_id', 'time'], inplace=True)
game_df['plus_minus'] = game_df.groupby('game_id')['home_diff'].diff()
unique_game_times = pd.DataFrame(np.unique(game_df[['game_id','time']], axis=0), columns = ['game_id','time'])\
                               .sort_values(['game_id','time'])
elapsed_time = unique_game_times.groupby(['game_id'])['time'].diff().shift(-1)
unique_game_times['elapsed_time'] = elapsed_time.divide(2880)
game_df = pd.merge(game_df, unique_game_times, on=['game_id' , 'time'], how='left')
team_df = pd.read_csv("/Users/martinbschmidt/Box/Basketball/Conferences/conference-2017_2018.csv")

team_df.columns = team_df.columns.str.lower()
team_df.head()
home_team = team_df.filter(['home_team','home_id', 'home_conf' ], axis=1)
away_team = team_df.filter(['away_team','away_id','away_conf'], axis=1)
game_conf_home_df = pd.merge(game_df, home_team, on='home_team', how='left')
game_conf_df = pd.merge(game_conf_home_df, away_team, on='away_team', how='left')
game_conf_df['home'] = 1
game_conf_df['away'] = 0
away_df = game_conf_df.drop(['home_team', 'home_player_id_1', 'home_player_id_2', 'home_player_id_3', 'home_player_id_4', 'home_player_id_5',\
                           'home_player_id_1_play_time', 'home_player_id_2_play_time', 'home_player_id_3_play_time', 'home_player_id_4_play_time', \
    
                           'home_player_id_5_play_time','home_id', 'home_conf','home'], axis=1)
home_df = game_conf_df.drop(['away_team', 'away_player_id_1', 'away_player_id_2', 'away_player_id_3', 'away_player_id_4', 'away_player_id_5',\
                           'away_player_id_1_play_time', 'away_player_id_2_play_time', 'away_player_id_3_play_time', 'away_player_id_4_play_time', \
                           'away_player_id_5_play_time','away_id', 'away_conf','away'], axis=1)
    
home_df.rename(columns={"home_team": "team","home_player_id_1": "player_id_1", "home_player_id_2": "player_id_2", "home_player_id_3": "player_id_3", \
                        "home_player_id_4": "player_id_4", "home_player_id_5": "player_id_5","home_player_id_1_play_time": "player_id_1_play_time", \
                        "home_player_id_2_play_time": "player_id_2_play_time", "home_player_id_3_play_time": "player_id_3_play_time", \
                        "home_player_id_4_play_time": "player_id_4_play_time", "home_player_id_5_play_time": "player_id_5_play_time","home_id": "team_id", \
                        "home_conf": "team_conf"}, inplace=True)
    
away_df.rename(columns={"away_team": "team","away_player_id_1": "player_id_1", "away_player_id_2": "player_id_2", "away_player_id_3": "player_id_3", 
                        "away_player_id_4": "player_id_4", "away_player_id_5": "player_id_5","away_player_id_1_play_time": "player_id_1_play_time", 
                        "away_player_id_2_play_time": "player_id_2_play_time", "away_player_id_3_play_time": "player_id_3_play_time", 
                        "away_player_id_4_play_time": "player_id_4_play_time", "away_player_id_5_play_time": "player_id_5_play_time","away_id": "team_id", 
                        "away_conf": "team_conf"}, inplace=True)

data = pd.concat([home_df, away_df], ignore_index=True)
data['home'].fillna(0, inplace=True)
data['away'].fillna(0, inplace=True)

strings = ['player_id_' + str(int(i+1)) for i in range(5)] 
one_stack_data = data.set_index(['game_id', 'plus_minus','team_id', 'time'])[strings].stack().reset_index()
colnames = one_stack_data.columns.values
colnames[-1] = 'player_id'
one_stack_data.columns = colnames
strings = ['player_id_' + str(int(i+1)) + '_play_time' for i in range(5)]
two_stack_data = data.set_index(['game_id','team_id'])[strings].stack().reset_index()
colnames = two_stack_data.columns.values
colnames[-1] = 'player_id_time'
two_stack_data.columns = colnames
two_stack_data.drop(['team_id', 'game_id'], axis=1, inplace=True)
stack_data = pd.concat([one_stack_data, two_stack_data], axis=1)
stack_data.drop(['level_2', 'level_4'], axis=1, inplace=True)
stack_data[np.isnan(stack_data)] = 0
stack_data.rename(columns={'player_id_time': 'min'}, inplace=True)
stack_data['year']=2018
year = 2018

teams= [1610612738, 1610612744, 1610612766, 1610612751, 1610612748,\
        1610612755, 1610612763, 1610612737, 1610612743, 1610612750,\
        1610612756, 1610612745, 1610612741, 1610612752, 1610612746,\
        1610612754, 1610612765, 1610612739, 1610612742, 1610612747,\
        1610612749, 1610612760, 1610612759, 1610612740, 1610612753,\
        1610612758, 1610612757, 1610612761, 1610612762, 1610612764]

for i in teams:
    team_df = stack_data.loc[stack_data['team_id'] == i ]
    save = ('/Users/martinbschmidt/Box/Basketball/Tanking/team_data/team_plus_minus_'+ str(i) + '_' + str(year) + '.csv' )
    team_df.to_csv(save)
    
    
