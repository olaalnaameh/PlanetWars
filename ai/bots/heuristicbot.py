import numpy
import random
import math
import time
import logging

from .. import planetwars_ai
from planetwars.datatypes import Order, ImmutableFleet
from planetwars.utils import *
from planetwars.heuristics import *
from ..state import State

# --- functions to adapt ---
def gships(pid, planet_data, cluster_data, t_dist, t_max, data, turn, train, planets, fleets, game_ratios):
    '''
    Calculates the number of ships the cluster needs to send in order to capture/help the planet
    @param pid: player id
    @type: int
    @param planet_data: the planet to send the ships to with data
                        e.g. planet_data['planet']  -  return the ImmutablePlanet object
    @type: {'planet' : p, 'value' : 0.0, 'value_d' : 0.0, 'influence_f' : 0.0, 'influence_e' : 0.0, 'influence_f_d' : 0.0, 'influence_e_d' : 0.0}
    @param cluster_data: Combined data of all planets that belong to this cluster
                         e.g. cluster_data['cluster']  -  return the cluster of [ImmutablePlanet] objects
    @type: {'cluster' : cluster, 'value' : 0.0, 'value_d' : 0.0, 'influence_f' : 0.0, 'influence_e' : 0.0, 'influence_f_d' : 0.0, 'influence_e_d' : 0.0}
    @param t_dist: turn distance between planet and cluster (approximation / take max)
    @type: int
    @param t_max: max turn distance on the map
    @type: int
    @param data: dictionary of all planets information. Each inner dictionary value is accesed using a planet as key
                 e.g.
                 data[key][planet.id][t_max]  -  key = 'value' | 'value_d'
                 data[key][planet.id]         -  key = 'influence_' + 'f' | 'e' | 'f_d' | 'e_d'
    @type: {'value' : {}, 'value_d' : {}, 'influence_f' : {}, 'influence_e' : {}, 'influence_f_d' : {}, 'influence_e_d' : {}}
    '''
    # input data sanity check
    assert isinstance(pid, int)
    assert isinstance(planet_data, dict)
    assert isinstance(cluster_data, dict)
    assert isinstance(t_dist, int)
    assert isinstance(t_max, int)
    assert isinstance(data, dict)
    # -------------------------------------------------------------------- 

    orders = []
    ships_needed_to_send = data['value'][planet_data['planet'].id][t_dist]
    all_ships = 0.0
    p_ships = {}
    for p in cluster_data['cluster']:
        all_ships += p.ships
        p_ships[p] = p.ships
    if all_ships < ships_needed_to_send:
        return orders

    percentage = ships_needed_to_send / all_ships

    for p in cluster_data['cluster']:
        orders.append(Order(p, planet_data['planet'], round(p.ships * percentage) + 1))

    # we need to return an array of Order objects. The array can be empty.
    return orders

# --- static functions ----
def select_move(turn, pid, planets, fleets, train):
    return compute_orders(turn, pid, planets, fleets, train, gships)

@planetwars_ai("Heuristic")
def agenttest_ai(turn, pid, planets, fleets, train):
    assert pid == 1 or pid == 2, "what?"
    return select_move(turn, pid, planets, fleets, train)
