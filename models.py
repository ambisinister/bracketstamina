import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

class Model:
    def roll(self, stamina, state=None):
        return random.uniform(0, stamina)

class Player:
    def __init__(self, stamina, model, name):
        self.stamina = stamina
        self.model = model
        self.name = name

    def roll(self, state=None):
        number = self.model.roll(self.stamina, state)
        self.stamina -= number
        return number

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def roll(self, stamina, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
            q_values = self(state_tensor)
            action_index = q_values.argmax().item()
            return stamina * (action_index / 99)

class DQNPlayer(Player):
    def __init__(self, stamina, dqn, name, num_actions=100, epsilon=0.1):
        super().__init__(stamina, dqn, name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epsilon = epsilon
        self.num_actions = num_actions

    def act(self, state):
        if random.random() < self.epsilon:
            # Choose a random action with probability epsilon
            action = random.randint(0, self.num_actions - 1)
            return self.stamina * (action / (self.num_actions - 1))
        else:
            # Choose the best action with probability 1 - epsilon
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.dqn(state_tensor.unsqueeze(0))
                action = torch.argmax(q_values).item()
                return self.stamina * (action / (self.num_actions - 1))

    def train(self, training_data, num_epochs, learning_rate, batch_size):
        memory = self._prepare_memory(training_data)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        best_loss = None

        for epoch in range(num_epochs):
            minibatches = self._sample_minibatches(memory, batch_size)
            epoch_loss = 0

            for ii, minibatch in enumerate(minibatches):
                states = [transition[0] for transition in minibatch]
                actions = [transition[1] for transition in minibatch]
                rewards = [transition[2] for transition in minibatch]
                next_states = [transition[3] for transition in minibatch]
                done_flags = [transition[4] for transition in minibatch]

                # Discretize the actions
                actions = [min(int(a / s * 100), 99) for a, s in zip(actions, [s[0] for s in states])]
                states = torch.tensor(states, dtype=torch.float32).to(self.device)
                actions = torch.tensor(actions, dtype=torch.long).to(self.device).view(-1, 1)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                done_flags = torch.tensor(done_flags, dtype=torch.bool).to(self.device)

                target_values = self.model(next_states).max(1)[0].detach()
                target_values[done_flags] = 0.0
                target_values = rewards + target_values
                predicted_values = self.model(states).gather(1, actions)

                #if ii % 1000 == 0:
                #  print([(x,y) for x,y in zip(predicted_values, target_values.unsqueeze(1))])

                loss = criterion(predicted_values, target_values.unsqueeze(1))
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if best_loss is None or epoch_loss < best_loss:
                torch.save(self.model, './best_weights.pth')
                best_loss = epoch_loss
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(minibatches):.4f}")

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.dqn(state_tensor).argmax().item()
        return action

    def _choose_action(self, state):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()
        self.model.train()
        percentage = action_idx / 100
        action = percentage * self.stamina
        return action

    def _prepare_memory(self, training_data):
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        memory = deque(maxlen=len(training_data))

        for data in training_data:
            memory.append(Transition(*data))

        return memory

    def _sample_minibatches(self, memory, batch_size):
        np.random.shuffle(memory)
        num_minibatches = len(memory) // batch_size
        minibatches = np.array_split(memory, num_minibatches)
        return minibatches    
