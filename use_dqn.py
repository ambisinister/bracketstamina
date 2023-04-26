import numpy as np
import matplotlib.pyplot as plt

from models import *
from bracket_functions import *

def use_dqn(model, stam, opp, rd, n):
    return dqn_player.roll(torch.tensor([stam, opp, rd, n]))

def visualize_policy(model, rd, n):
    stam_values = np.linspace(0, 15, 25)
    opp_values = np.linspace(0, 15, 25)
    policy_matrix = np.zeros((len(stam_values), len(opp_values)))

    for i, stam in enumerate(stam_values):
        for j, opp in enumerate(opp_values):
            rd_roll = use_dqn(model, stam, opp, rd, n)
            model.stamina += rd_roll #annoying implementation detail
            policy_matrix[i, j] = rd_roll

    plt.imshow(policy_matrix, cmap="viridis", origin="lower", extent=[0, 15, 0, 15], aspect='auto')
    plt.xlabel("Opponent's stamina")
    plt.ylabel("Player's stamina")
    plt.title(f"Policy Visualization at Round {rd}")
    plt.colorbar(label="Output value")
    plt.savefig(f'./figures/policy_{rd}.png')
    plt.clf()

if __name__ == '__main__':
    # Load the DQNPlayer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = torch.load('./dqn_weights.pth')
    dqn_player = DQNPlayer(15, dqn.to(device), "DQN Player")

    for i in range(4):
        visualize_policy(dqn_player, i, 16)
    #visualize_policy(dqn_player, 4, 16)
