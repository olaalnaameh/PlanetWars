import numpy
import random
import math
import time
import logging

from .. import planetwars_ai
from planetwars.datatypes import Order, ImmutablePlanet, ImmutableFleet
from planetwars.utils import *
from planetwars.heuristics import *
from ..state import State
from planetwars.neuralnetwork import NNetwork
from theano import config
config.floatX = "float32"

# --- functions to adapt ---
def gships(pid, planet_data, clustr_data, t_dist, t_max, data, turn, train, planets, fleets, game_metrics):
    '''
    Calculates the number of ships the cluster needs to send in order to capture/help the planet
    @param pid: player id
    @type: int
    @param planet_data: the planet to send the ships to with data
    @type: {'planet' : p, 'value' : 0.0, 'value_d' : 0.0, 'influence_f' : 0.0, 'influence_e' : 0.0, 'influence_f_d' : 0.0, 'influence_e_d' : 0.0}
    @param cluster_data: Combined data of all planets that belong to this cluster
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
    assert isinstance(clustr_data, dict)
    assert isinstance(t_dist, int)
    assert isinstance(t_max, int)
    assert isinstance(data, dict)
    # -------------------------------------------------------------------- 
    # create relation value 1 / 0.5 / 0 depending of the planet is neutral / friendly / enemy
    cluster_planet_relationship = 0.0
    if pid == planet_data['planet'].owner:
        cluster_planet_relationship = 0.5
    elif planet_data['planet'].owner == 0:
        cluster_planet_relationship = 1



    # +1 to avoid division by 0
    # we send data as ratios (planet under cluster)
    # all values are between 0 and 1; except "owner" and "t_dist"
    input = [# cluster data
             clustr_data['ships'],
             clustr_data['growth'],
             clustr_data['value'],
             #clustr_data['value_d'],
             clustr_data['influence_f'],
             clustr_data['influence_e'],
             clustr_data['influence_f_m'],
             clustr_data['influence_e_m'],
             #clustr_data['influence_f_md'],
             #clustr_data['influence_e_md'],
             # planet data
             planet_data['ships'],
             planet_data['growth'],
             planet_data['value'],
             #planet_data['value_d'],
             planet_data['influence_f'],
             planet_data['influence_e'],
             planet_data['influence_f_m'],
             planet_data['influence_e_m'],
             #planet_data['influence_f_md'],
             #planet_data['influence_e_md'],
             game_metrics['growth_r'],       # overall growth ratio
             game_metrics['growth_er'],      # overall growth ratio enemy
             game_metrics['ships_r'],        # overall ship ratio
             cluster_planet_relationship,    # read above for details 
             t_dist / float(t_max)]
    
    assert not any(numpy.isnan(in_) or numpy.isinf(in_) for in_ in input)
    input_np = numpy.array(input, ndmin=2, dtype=config.floatX)

    # two output values. 
    # 1. The certainty of this attack / help
    # 2. the percentage og ships to send from the cluster
    out_np = NNetwork.gships.predict(input_np, batch_size = 1)
    out = out_np[0][0] # reformat the output data    

    assert not (numpy.isnan(out) or numpy.isinf(out))

    orders = []
    for p in clustr_data['cluster']:
        orders.append(Order(p, planet_data['planet'], int(p.ships / 2)))

    # if we are training, add pattern
    pattern = None
    if train:
        pattern = {
            'in' : input_np, 
            'out' : out_np,
            'reward' : 0.0,
            'in_' : None,
            'terminal' : False}       

    # we need to return an array of Order objects. The array can be empty.
    return (out, orders, pattern, t_dist, planet_data['value'])

# --- static functions ----
def select_move(turn, pid, planets, fleets, train):
    return compute_orders(turn, pid, planets, fleets, train, gships)

@planetwars_ai("HeuristicNN")
def agenttest_ai(turn, pid, planets, fleets, train):
    assert pid == 1 or pid == 2, "what?"    
    return select_move(turn, pid, planets, fleets, train)

