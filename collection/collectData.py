from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
import pandas as pd

def main():
	boxscore = getBoxScore()

'''
Returns the traditional box scores season averages for each player per 36 minutes
Only keeps players that played more than 1000 minutes 
'''
def getBoxScore(season='2018-19', MIN_MINUTES_PLAYED=1000):
	init = False
	rawBoxScores = None
	playerIDs = getPlayers()[:35]
	for pid in playerIDs:
		print(pid)
		careerStats = None
		try:
			careerStats = playercareerstats.PlayerCareerStats(pid).season_totals_regular_season.get_data_frame()
		except Exception as e:
			continue
		seasonStats = careerStats.loc[careerStats['SEASON_ID'] == season]
		seasonStats = seasonStats.loc[seasonStats['MIN'] >= MIN_MINUTES_PLAYED]
		if seasonStats.count()[0] > 0:
			if not init:
				rawBoxScores = seasonStats
				init = True
			else:
				rawBoxScores = rawBoxScores.append(seasonStats)

	# Normalize stats by per 36 MIN
	rawBoxScores = rawBoxScores[['PLAYER_ID','MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']]
	normedBoxScores = pd.DataFrame(columns=rawBoxScores.columns)

	toCopy = ['PLAYER_ID', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
	for c in toCopy:
		normedBoxScores[c] = rawBoxScores[c]

	normedMins = rawBoxScores['MIN'] / 36
	for c in rawBoxScores.columns:
		if c not in toCopy:
			normedBoxScores[c] = rawBoxScores[c] / normedMins

	return normedBoxScores

'''
Returns the player IDs of all active NBA players
'''
def getPlayers():
	activePlayers = players.get_active_players()
	ids = [player['id'] for player in activePlayers]
	return ids


if __name__ == '__main__':
	main()