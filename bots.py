import numpy as np

from models import *
from bracket_functions import *

if __name__ == '__main__':
    try:
        # Load the DQNPlayer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dqn = torch.load('./dqn_weights.pth')
        dqn_player = DQNPlayer(15, dqn.to(device), "DQN Player")
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dqn_player = DQNPlayer(12, Model(), "DQN Player", device)

        num_training_tournaments = 15000
        num_players = 16
        training_data = generate_training_data(num_training_tournaments, num_players, dqn_player.stamina)

        # Train the DQNPlayer
        dqn_player = DQNPlayer(12, DQN(4, 100, 8).to(device), "DQN Player")
        num_epochs = 100
        learning_rate = 3e-7
        batch_size = 64
        dqn_player.train(training_data, num_epochs, learning_rate, batch_size)

        torch.save(dqn_player, './dqn_model.pth')

    # Test the DQNPlayer
    num_tournaments = 3000
    #players = [Player(i + 1, Model(), f"Player {i + 1}") for i in range(16)]
    players = [Player(15, Model(), f"Player {i + 1}") for i in range(16)]
    players[-1] = dqn_player
    #players[-1] = Player(22, Model(), "Player of interest")
    print([x.stamina for x in players])

    probabilities = run_tournaments(num_tournaments, players)
    
    for player, probability in probabilities.items():
        print(f"{player}'s probability to win: {probability:.4f}")
