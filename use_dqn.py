import numpy as np
import matplotlib.pyplot as plt

#from models import *
#from bracket_functions import *

def use_dqn(model, stam, opp, rd, n):
    return dqn_player.roll(torch.tensor([stam, opp, rd, n]))

def visualize_policy(model, rd, n):
    stam_values = np.linspace(0, 15, 15)
    opp_values = np.linspace(0, 15, 15)
    policy_matrix = np.zeros((len(stam_values), len(opp_values)))

    for i, stam in enumerate(stam_values):
        for j, opp in enumerate(opp_values):
            policy_matrix[i, j] = use_dqn(model, stam, opp, rd, n)

    plt.imshow(policy_matrix, cmap="viridis", origin="lower", extent=[0, 15, 0, 15], aspect='auto')
    plt.xlabel("Opponent's stamina")
    plt.ylabel("Player's stamina")
    plt.title(f"Policy Visualization at Round {rd}")
    plt.colorbar(label="Output value")
    plt.show()

if __name__ == '__main__':
    # Load the DQNPlayer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = torch.load('./dqn_weights.pth')
    dqn_player = DQNPlayer(15, dqn.to(device), "DQN Player")

    print(use_dqn(dqn_player, 12, 11, 4, 16)) 
    #visualize_policy(dqn_player, 4, 16)
