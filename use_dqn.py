import numpy as np

from models import *
from bracket_functions import *

if __name__ == '__main__':
    # Load the DQNPlayer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = torch.load('./dqn_weights.pth')
    dqn_player = DQNPlayer(15, dqn.to(device), "DQN Player")

    num_training_tournaments = 1
    num_players = 16
    training_data = generate_training_data(num_training_tournaments, num_players, dqn_player.stamina)

    stamina = 15
    opp_stam = 4
    which_round = 0
    n_players = 16

    print(dqn_player.roll(torch.tensor([stamina, opp_stam, which_round, n_players])))
