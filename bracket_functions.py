import copy
import random
from math import log2
import collections
from collections import namedtuple, deque
import numpy as np

from models import *

class SingleEliminationBracket:
    def __init__(self, players):
        self.players = sorted(players, key=lambda x: x.stamina, reverse=True)
        self.bracket = self._initialize_bracket()
        self.match_history = {player: [] for player in self.players}

    def _initialize_bracket(self):
        num_players = len(self.players)
        num_rounds = int(log2(num_players))

        bracket = [[] for _ in range(num_rounds+1)]
        bracket[0] = self.players

        return bracket

    def play_tournament(self):
        for round_idx in range(len(self.bracket)-1):
            self._play_round(round_idx)

        return self.bracket[-1][0]

    def _play_round(self, round_idx):
        winners = []
        players = self.bracket[round_idx]
        left, right = 0, len(players) - 1

        while left < right:
            p1 = players[left]
            p2 = players[right]
            state = (p1.stamina, p2.stamina, round_idx, len(self.players))

            p1_roll = p1.roll(state)
            p2_roll = p2.roll(state)

            if p1_roll >= p2_roll:
                winner, loser = p1, p2
            else:
                winner, loser = p2, p1

            self.match_history[p1].append({'state': state, 'roll': p1_roll, 'next_state': (winner.stamina, loser.stamina, round_idx + 1, len(self.players)), 'done': p1 == loser})
            self.match_history[p2].append({'state': state, 'roll': p2_roll, 'next_state': (winner.stamina, loser.stamina, round_idx + 1, len(self.players)), 'done': p2 == loser})

            winners.append(winner)

            left += 1
            right -= 1

        self.bracket[round_idx + 1] = winners

def run_tournaments(num_tournaments, players):
    model = Model()
    winners = []

    original_players = players

    for _ in range(num_tournaments):
        bracket = SingleEliminationBracket(copy.deepcopy(original_players))
        winner = bracket.play_tournament()
        winners.append(winner.name)

    counter = collections.Counter(winners)
    probabilities = {player: count / num_tournaments for player, count in counter.items()}
    
    return probabilities
