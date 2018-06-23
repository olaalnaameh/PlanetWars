from collections import defaultdict, Sequence
from math import ceil, sqrt
import numpy
import random
import logging
import time
from planetwars.datatypes import ImmutableFleet, ImmutablePlanet, Order
from planetwars.utils import turn_dist, aggro_partition
logging.basicConfig(filename='log.log',level=logging.INFO)

random_move = 0.10
initial_growth_ratio = 0.0

# --- static functions ---
def get_max(dic):
    '''
    Gets the maximum of a dict data. Specifically made to work with dictionaries
    @param dic: dictionary of floats
    @type: {}
    '''
    # input data sanity check
    assert isinstance(dic, dict)

    return max([numpy.abs(max(dic.iteritems(), key=lambda x: x[1])[1]), 
                numpy.abs(min(dic.iteritems(), key=lambda x: x[1])[1])])
def compute_dist_and_t_max(planets):
    '''
    Compute all distances between all planets
    Returns a tuple of a dictionary of planet ids [id][id] gives the distance between those 2 planets
    and the max distance between planets that can exist t_max
    @param planets:
    @type: [ImmutablePlanet]
    '''
    # input data sanity check
    assert isinstance(planets, Sequence)
    if len(planets) > 0:
        assert isinstance(planets[0], ImmutablePlanet)

    distances = {}
    t_max = 0
    for p in planets:
        dis = {}
        for p_ in planets:
            dis[p_.id] = int(turn_dist(p,p_))
            if t_max < dis[p_.id]:
                t_max = dis[p_.id]
        distances[p.id] = dis

    return distances, int(t_max)
def compute_static_data(pid, planets, t_max):
    '''
    Compute all distances between all planets
    Returns a tuple of a dictionary of planet ids [id][id] gives the distance between those 2 planets
    and the max distance between planets that can exist t_max
    @param planets:
    @type: [ImmutablePlanet]
    '''
    # input data sanity check
    assert isinstance(planets, Sequence)
    assert isinstance(t_max, int)

    fleets_dict = { 0 : {}, 1 : {}, 2 : {}}
    bucket = [0.0] * t_max
    my_planets = []
    fleets_from_this_planet = {}
    for p in planets:
        fleets_from_this_planet[p.id] = False
        fleets_dict[0][p.id] = bucket[:]
        fleets_dict[1][p.id] = bucket[:]
        fleets_dict[2][p.id] = bucket[:]
        if p.owner == pid:
            my_planets.append(p)
    return fleets_dict, my_planets, fleets_from_this_planet
def compute_desired_growth(pid, turn):
    global initial_growth_ratio
    return (1 + initial_growth_ratio - 130**(-turn/800.0))
def compute_game_growth_ships_ratios(pid, turn, planets, fleets):
    '''
    Computes growth and ship ratio for the current game state
    '''
    # get enemy id
    eid = 1
    if pid == eid:
        eid = 2

    growth_ = [0.0, 0.0, 0.0]
    ships_ = [0.0, 0.0, 0.0]
    for p in planets:
        ships_[p.owner] += p.ships
        growth_[p.owner] += p.growth
    for f in fleets:
        ships_[f.owner] += f.ships    

    if growth_[eid] == 0.0 or growth_[pid] == 0.0 or ships_[eid] == 0 or ships_[pid] == 0:
        return None

    if turn == 0: 
        global initial_growth_ratio
        initial_growth_ratio = growth_[pid] / (growth_[pid] + growth_[eid] + growth_[0])

    return {'growth_us' : growth_[pid],
            'growth_e' : growth_[eid],
            'growth_r' : growth_[pid] / (growth_[pid] + growth_[eid] + growth_[0]),
            'growth_er' : growth_[eid] / (growth_[pid] + growth_[eid] + growth_[0]),
            'ships_us' : ships_[pid],
            'ships_e' : ships_[eid],
            'ships_r' : ships_[pid] / (ships_[pid] + ships_[eid]),
            'growth_desired' : compute_desired_growth(pid, turn)}
def init_data(t_max):
    '''

    '''
    data = {'value' : {}, 'value_max' : {}, 'value_min' : {},
            'value_d' : {}, 'value_d_max' : {}, 'value_d_min' : {},
            'influence_f' : {}, 'influence_f_max' : {}, 'influence_f_min' : {},
            'influence_e' : {}, 'influence_e_max' : {}, 'influence_e_min' : {},
            'influence_f_m' : {}, 'influence_f_m_max' : {}, 'influence_f_m_min' : {},
            'influence_e_m' : {}, 'influence_e_m_max' : {}, 'influence_e_m_min' : {},
            'influence_f_md' : {}, 'influence_f_md_max' : {}, 'influence_f_md_min' : {},
            'influence_e_md' : {}, 'influence_e_md_max' : {}, 'influence_e_md_min' : {}}

    for i in range(0, t_max + 1):
        data['value_max'][i] = - numpy.inf
        data['value_d_max'][i] = - numpy.inf
        data['value_min'][i] = numpy.inf
        data['value_d_min'][i] = numpy.inf

    data['influence_f_max'] = - numpy.inf
    data['influence_e_max'] = - numpy.inf
    data['influence_f_m_max'] = - numpy.inf
    data['influence_e_m_max'] = - numpy.inf
    data['influence_f_md_max'] = - numpy.inf
    data['influence_e_md_max'] = - numpy.inf
        
    data['influence_f_min'] = numpy.inf
    data['influence_e_min'] = numpy.inf
    data['influence_f_m_min'] = numpy.inf
    data['influence_e_m_min'] = numpy.inf
    data['influence_f_md_min'] = numpy.inf
    data['influence_e_md_min'] = numpy.inf

    return data
def compute_data_norm(data, game_metrics, t_max):
    data_norm =  {}  
    data_norm['ships'] = game_metrics['ships_us'] + game_metrics['ships_e']
    data_norm['growth'] = game_metrics['growth_us'] + game_metrics['growth_e']
    data_norm['value'] = data['value_max'][t_max] - data['value_min'][t_max]
    data_norm['value_d'] = data['value_d_max'][t_max] - data['value_d_min'][t_max]  
    data_norm['influence_f'] = data['influence_f_max'] - data['influence_f_min'] 
    data_norm['influence_e'] = data['influence_e_max'] - data['influence_e_min']
    data_norm['influence_f_m'] = data['influence_f_m_max'] - data['influence_f_m_min'] 
    data_norm['influence_e_m'] = data['influence_e_m_max'] - data['influence_e_m_min']
    data_norm['influence_f_md'] = data['influence_f_md_max'] - data['influence_f_md_min']
    data_norm['influence_e_md'] = data['influence_e_md_max'] - data['influence_e_md_min']
    for key in data_norm:
        if data_norm[key] == 0.0:
            data_norm[key] = 1.0 # we do this to prevent division by 0    

    return data_norm
def update_influence_min_max(data, id):
    # get max
    data['influence_f_max'] = max(data['influence_f_max'], data['influence_f'][id])
    data['influence_f_m_max'] = max(data['influence_f_m_max'], data['influence_f_m'][id])
    data['influence_f_md_max'] = max(data['influence_f_md_max'], data['influence_f_md'][id])
    data['influence_e_max'] = max(data['influence_e_max'], data['influence_e'][id])
    data['influence_e_m_max'] = max(data['influence_e_m_max'], data['influence_e_m'][id])
    data['influence_e_md_max'] = max(data['influence_e_md_max'], data['influence_e_md'][id])

    # get min
    data['influence_f_min'] = min(data['influence_f_min'], data['influence_f'][id])
    data['influence_f_m_min'] = min(data['influence_f_m_min'], data['influence_f_m'][id])
    data['influence_f_md_min'] = min(data['influence_f_md_min'], data['influence_f_md'][id])
    data['influence_e_min'] = min(data['influence_e_min'], data['influence_e'][id])
    data['influence_e_m_min'] = min(data['influence_e_m_min'], data['influence_e_m'][id])
    data['influence_e_md_min'] = min(data['influence_e_md_min'], data['influence_e_md'][id])
def planet_value(planet, fleets_f, fleets_e, t_max, d, sgn, data):
    '''
    Returns the value of a planet over t turns. Positive planet value signifies that the planet will be kept by current owner
    Important! fleets must be already seperated into friendly/enemy depending on the planet type
    @param planet:
    @type: ImmutablePlanet
    @param fleets:
    @type: [ImmutableFleet]
    @param t: number of relevant turns
    @type: int
    @param d: decay parameter
    @type: int
    '''
    # input data sanity check
    assert isinstance(planet, ImmutablePlanet)
    assert isinstance(t_max, int)
    assert isinstance(d, float)

    # init data
    g = float(planet.growth)
    value = float(planet.ships)
    value_d = float(planet.ships)
    data['value_max'][0] = max(data['value_max'][0], value)
    data['value_d_max'][0] = max(data['value_d_max'][0], value_d)
    data['value_min'][0] = min(data['value_min'][0], value)
    data['value_d_min'][0] = min(data['value_d_min'][0], value_d)

    values = [ value ]
    values_d = [ value ]
    friendly_f = 0.0
    enemy_f = 0.0
    for i in range(1, t_max + 1):
        # calculate planet Value without/with decay for each t and add that to an array
        if fleets_f != None:
            friendly_f = fleets_f[planet.id][i - 1] 
        if fleets_e != None:
            enemy_f = fleets_e[planet.id][i - 1]
        dif = (friendly_f - enemy_f)
        value += sgn(value) * g + dif
        value_d += sgn(value_d) * g + (dif + 1) / (d * numpy.log2(i + 1) + 1) - 1
        values.append(value)
        values_d.append(value_d)
        data['value_max'][i] = max(data['value_max'][i], value)
        data['value_d_max'][i] = max(data['value_d_max'][i], value_d)
        data['value_min'][i] = min(data['value_min'][i], value)
        data['value_d_min'][i] = min(data['value_d_min'][i], value_d)

    return values, values_d
def planet_external_influence(planet, planets, t_max, distances, d, data):
    '''
    Returns the external influence of all other planet values in relation to this planet. 
    The relation is defined by the distance between the planets and the same decay parameter.
    By distance, I mean the number of turns for a fleet to reach from planet B to A, where A 
    is the main planet we are considering.
    @param planet:
    @type: ImmutablePlanet
    @param planets:
    @type: [ImmutablePlanet]
    @param fleets:
    @type: [ImmutableFleet]
    @param d: decay parameter
    @type: float
    '''
    # input data sanity check
    assert isinstance(planet, ImmutablePlanet)
    assert isinstance(d, float)
    assert isinstance(planets, Sequence)
    assert isinstance(distances, dict)
    assert isinstance(data, dict)

    influence = 0.0
    influence_m = 0.0
    influence_md = 0.0
    l = max(len(planets), 1)
    for p in planets:
        if planet.id == p.id:
            continue
        influence += data['value'][p.id][1] / distances[planet.id][p.id]
        influence_m += (data['value'][p.id][1] + data['value'][p.id][t_max]) / (2 * distances[planet.id][p.id])
        influence_md += (data['value_d'][p.id][1] + data['value_d'][p.id][t_max]) / (2 * distances[planet.id][p.id])
    return influence / l, influence_m / l, influence_md / l
def find_closest_planet_turn_dist(planet, planets, distances):
    '''
    Finds the closest planet turn distance in relation to the specified planet. 
    @param planet: the planet for which we want to find the closest other planet
    @type: ImmutablePlanet
    @param planets: all planets to search for the distance
    @type: [ImmutablePlanet]
    @param distances: Dictionary of [p1.id][p2.id] that returns the number of turn distance between two planets, based on id
    @type: {int, {int, int}}
    '''
    # input data sanity check
    assert isinstance(planet, ImmutablePlanet)
    assert isinstance(planets, Sequence)
    assert isinstance(distances, dict)
    if len(planets) > 0:
        assert isinstance(planets[0], ImmutablePlanet)

    filter = [distances[planet.id][p.id] for p in planets if planet.id != p.id]
    if len(filter) > 0:
        return min(filter)
    else:
        return 0.0
def associate_clusters(planets, planets_c, distances, cluster_outer_radius):
    '''
    Returns the specified planet with its associated cluster influential of planets.
    By influential, we mean the planets with the closest distance combined with the cluster_outer_radius.
    We also exclude the specified planet from being part of the cluster
    @param planet: the planet for which we want to find the closest other planet
    @type: ImmutablePlanet
    @param planets: all planets to search for the distance
    @type: [ImmutablePlanet]
    @param distances: Dictionary of [p1.id][p2.id] that returns the number of turn distance between two planets, based on id
    @type: {int, {int, int}}
    @param cluster_radius: the turn distance between the determined closest planet turn distance and it's outer radius
    @type: int
    '''
    # input data sanity check
    assert isinstance(planets, Sequence)
    assert isinstance(planets_c, Sequence)
    assert isinstance(distances, dict)
    assert isinstance(cluster_outer_radius, int)

    output = []
    for p in planets:
        relevant_turn_radius = find_closest_planet_turn_dist(p, planets_c, distances)
        if relevant_turn_radius == 0: # we found ourself / same planet. So we have no other friends
            continue 
        else:
            relevant_turn_radius += cluster_outer_radius
        cluster = [p_ for p_ in planets_c if (p.id != p_.id and distances[p.id][p_.id] <= relevant_turn_radius)]

        # if no relevant nearby cluster, we take the next weak planet
        if len(cluster) == 0:
            continue

        # get max distance between planet and associated planets
        # IMPORTANT: we must choose max between these two because of some future normalization of data to make sense!
        turn_dist_max = max([int(turn_dist(p, p_)) for p_ in cluster])
        output.append((p, cluster, turn_dist_max))    
    return output
def compute_data(pid, planets, fleets_dict, distances, t_max, d, cluster_outer_radius, game_metrics):
    '''
    Computes the data for all planets. Value functions use t_max.
    @param pid: player id
    @type: int
    @param planets: all planets
    @type: [ImmutablePlanet]
    @param planets: all fleets
    @type: [ImmutableFleet]
    @param distances: Dictionary of [p1.id][p2.id] that returns the number of turn distance between two planets, based on id
    @type: {int, {int, int}}
    @param t_max: t_max
    @type: int
    @param d: decay parameter
    @type: float
    @param cluster_radius: the turn distance between the determined closest planet turn distance and it's outer radius
    @type: int
    @param data: dictionary of all planets information. Each inner dictionary value is accesed using a planet as key
    @type: {'value' : {}, 'value_d' : {}, 'influence_f' : {}, 'influence_e' : {}, 'influence_f_d' : {}, 'influence_e_d' : {}}
    '''
    # input data sanity check
    assert isinstance(pid, int)
    assert isinstance(planets, Sequence)
    assert isinstance(distances, dict)
    assert isinstance(t_max, int)
    assert isinstance(d, float)
    assert isinstance(cluster_outer_radius, int)    

    # get enemy id. 0 is neutral always
    eid = 2
    if pid == eid:
        eid = 1    

    # init data structure
    data = init_data(t_max)

    # split data into relevant categories
    my_planets, their_planets, neutral_planets = aggro_partition(pid, planets)

    # calculate planet values based on all t values without/with decay  
    sgn = lambda a: (a>0) - (a<0) # function used to determine the sign of a number
    for p in planets:
        if pid == p.owner: # our perspective
            fleets_f, fleets_e = fleets_dict[pid], fleets_dict[eid]
        elif p.owner == 0: # neutral perspective
            fleets_f, fleets_e = None, fleets_dict[0]
        else: # enemy perspective
            fleets_f, fleets_e = fleets_dict[eid], fleets_dict[pid]
        data['value'][p.id], data['value_d'][p.id] = planet_value(p, fleets_f, fleets_e, t_max, d, sgn, data)

    # calculate the influence of other planets to each planet (friendly / enemy) without/with decay
    # also get all associated clusters. We do this after computing values
    # we have 3 for loops, but we slip the planets into categories, basically we have one for loop for efficiency.
    for p in my_planets: # first we consider our planets
        data['influence_f'][p.id], data['influence_f_m'][p.id], data['influence_f_md'][p.id] = planet_external_influence(p, my_planets, t_max, distances, d, data)
        data['influence_e'][p.id], data['influence_e_m'][p.id], data['influence_e_md'][p.id] = planet_external_influence(p, their_planets, t_max, distances, d, data)
        update_influence_min_max(data, p.id)
    for p in their_planets: # now the enemy, here we switch what represents "my_planets" and "their_planets"
        data['influence_f'][p.id], data['influence_f_m'][p.id], data['influence_f_md'][p.id] = planet_external_influence(p, their_planets, t_max, distances, d, data)
        data['influence_e'][p.id], data['influence_e_m'][p.id], data['influence_e_md'][p.id] = planet_external_influence(p, my_planets, t_max, distances, d, data)
        update_influence_min_max(data, p.id)
    for p in neutral_planets: # now the neutral, here we add "my_planets" and "their_planets". Neutral planets cannot have friendly influence
        data['influence_f'][p.id], data['influence_f_m'][p.id], data['influence_f_md'][p.id] = 0.0 , 0.0, 0.0
        data['influence_e'][p.id], data['influence_e_m'][p.id], data['influence_e_md'][p.id] = planet_external_influence(p, my_planets + their_planets, t_max, distances, d, data)
        update_influence_min_max(data, p.id)   

    # return all dictionaries with data
    return data, compute_data_norm(data, game_metrics, t_max)
def compute_cluster_data(planet, cluster, t_dist, distances, d, cluster_outer_radius, data, data_norm):
    '''
    @param p: the planet to compute cluster & correlated data for
    @type: ImmutablePlanet
    @param my_planets_s: all of my strong planets
    @type: [ImmutablePlanet]
    @param planets: all fleets
    @type: [ImmutableFleet]
    @param distances: Dictionary of [p1.id][p2.id] that returns the number of turn distance between two planets, based on id
    @type: {int, {int, int}}
    @param d: decay parameter
    @type: float
    @param cluster_radius: the turn distance between the determined closest planet turn distance and it's outer radius
    @type: int
    @param data: dictionary of all planets information. Each inner dictionary value is accesed using a planet as key
    @type: {'value' : {}, 'value_d' : {}, 'influence_f' : {}, 'influence_e' : {}, 'influence_f_d' : {}, 'influence_e_d' : {}}
    '''
    # input data sanity check
    assert isinstance(planet, ImmutablePlanet)
    assert isinstance(cluster, Sequence)
    assert isinstance(distances, dict)
    assert isinstance(d, float)
    assert isinstance(cluster_outer_radius, int)
    assert isinstance(data, dict)

    # get associated clusters for planet p_w. Clusters must contain only our strong planets
    clustr_data = {'cluster' : cluster, 
                      'ships' : 0.0, # we have ships and growth here which is the sum for all planets in the cluster
                      'growth' : 0.0,
                      'value' : 0.0, 
                      'value_d' : 0.0, 
                      'influence_f' : 0.0, 
                      'influence_e' : 0.0, 
                      'influence_f_m' : 0.0, 
                      'influence_e_m' : 0.0,
                      'influence_f_md' : 0.0, 
                      'influence_e_md' : 0.0}
    planet_data = {'planet' : planet, 
                'ships' : planet.ships,
                'growth' : planet.growth,
                'value' : 0.0, 
                'value_d' : 0.0, 
                'influence_f' : 0.0, 
                'influence_e' : 0.0, 
                'influence_f_m' : 0.0, 
                'influence_e_m' : 0.0,
                'influence_f_md' : 0.0, 
                'influence_e_md' : 0.0}    

    # compute common growth and ships for the planets in cluster
    clustr_data['ships'] = numpy.sum([p_.ships for p_ in clustr_data['cluster']])
    clustr_data['growth'] = numpy.sum([p_.growth for p_ in clustr_data['cluster']])

    # compute Value based on turn_dist (with decay) for planet p & cluster (we combine data for the cluster)      
    # p_output['value'] = data['value'][p.id][t_dist]
    # p_output['value_d'] =  data['value_d'][p.id][t_dist]
    planet_data['value'] = numpy.mean([data['value'][planet.id][int(turn_dist(planet, p_))] for p_ in clustr_data['cluster']])
    planet_data['value_d'] =  numpy.mean([data['value_d'][planet.id][int(turn_dist(planet, p_))] for p_ in clustr_data['cluster']])
    clustr_data['value'] = numpy.sum([data['value'][p_.id][int(turn_dist(planet, p_))] for p_ in clustr_data['cluster']])
    clustr_data['value_d'] = numpy.sum([data['value_d'][p_.id][int(turn_dist(planet, p_))] for p_ in clustr_data['cluster']])

    # get relevant influence and mean() all the influence for the cluster planets
    # no aditional computation is needed, because the influence is calculated with associated t (distance)
    clustr_data['influence_f'] = numpy.mean([data['influence_f'][p_.id] for p_ in clustr_data['cluster']])
    clustr_data['influence_e'] = numpy.mean([data['influence_e'][p_.id] for p_ in clustr_data['cluster']])
    clustr_data['influence_f_m'] = numpy.mean([data['influence_f_m'][p_.id] for p_ in clustr_data['cluster']])
    clustr_data['influence_e_m'] = numpy.mean([data['influence_e_m'][p_.id] for p_ in clustr_data['cluster']])
    clustr_data['influence_f_md'] = numpy.mean([data['influence_f_md'][p_.id] for p_ in clustr_data['cluster']])
    clustr_data['influence_e_md'] = numpy.mean([data['influence_e_md'][p_.id] for p_ in clustr_data['cluster']])

    # copy from data for easy reference
    planet_data['influence_f'] = data['influence_f'][planet.id]
    planet_data['influence_e'] = data['influence_e'][planet.id]
    planet_data['influence_f_m'] = data['influence_f_m'][planet.id]
    planet_data['influence_e_m'] = data['influence_e_m'][planet.id]
    planet_data['influence_f_md'] = data['influence_f_md'][planet.id]
    planet_data['influence_e_md'] = data['influence_e_md'][planet.id]

    # we update normed data for t_dist instead of t_max
    data_norm['value'] = data['value_max'][t_dist] - data['value_min'][t_dist]
    data_norm['value_d'] = data['value_d_max'][t_dist] - data['value_d_min'][t_dist]
    # adjust for division by 0
    if data_norm['value'] == 0.0:
        data_norm['value'] = 1.0
    if data_norm['value_d'] == 0.0:
        data_norm['value_d'] = 1.0

    # normalize all data!! ships / growth use normal normalization; the rest use min/max normalization
    # data_norm will return 1 in case of 0, this means that we prevent division by 0
    planet_data['ships'] =           planet_data['ships']                                           / data_norm['ships']
    planet_data['growth'] =          planet_data['growth']                                          / data_norm['growth'] 
    planet_data['value'] =          (planet_data['value']           - data['value_min'][t_dist])    / data_norm['value']
    planet_data['value_d'] =        (planet_data['value_d']         - data['value_d_min'][t_dist])  / data_norm['value_d']
    planet_data['influence_f'] =    (planet_data['influence_f']     - data['influence_f_min'])      / (data_norm['influence_f'] + data_norm['influence_e'])
    planet_data['influence_e'] =    (planet_data['influence_e']     - data['influence_e_min'])      / (data_norm['influence_e'] + data_norm['influence_f'])
    planet_data['influence_f_m'] =  (planet_data['influence_f_m']   - data['influence_f_m_min'] )   / (data_norm['influence_f_m'] + data_norm['influence_e_m'])
    planet_data['influence_e_m'] =  (planet_data['influence_e_m']   - data['influence_e_m_min'])    / (data_norm['influence_e_m'] + data_norm['influence_f_m'])
    planet_data['influence_f_md'] = (planet_data['influence_f_md']  - data['influence_f_md_min'])   / (data_norm['influence_f_md'] + data_norm['influence_e_md'])
    planet_data['influence_e_md'] = (planet_data['influence_e_md']  - data['influence_e_md_min'])   / (data_norm['influence_e_md'] + data_norm['influence_f_md'])
    clustr_data['ships'] =           clustr_data['ships']                                           / data_norm['ships']
    clustr_data['growth'] =          clustr_data['growth']                                          / data_norm['growth'] 
    clustr_data['value'] =          (clustr_data['value']           - data['value_min'][t_dist])    / data_norm['value']
    clustr_data['value_d'] =        (clustr_data['value_d']         - data['value_d_min'][t_dist])  / data_norm['value_d']
    clustr_data['influence_f'] =    (clustr_data['influence_f']     - data['influence_f_min'])      / (data_norm['influence_f'] + data_norm['influence_e'])
    clustr_data['influence_e'] =    (clustr_data['influence_e']     - data['influence_e_min'])      / (data_norm['influence_e'] + data_norm['influence_f'])
    clustr_data['influence_f_m'] =  (clustr_data['influence_f_m']   - data['influence_f_m_min'] )   / (data_norm['influence_f_m'] + data_norm['influence_e_m'])
    clustr_data['influence_e_m'] =  (clustr_data['influence_e_m']   - data['influence_e_m_min'])    / (data_norm['influence_e_m'] + data_norm['influence_f_m'])
    clustr_data['influence_f_md'] = (clustr_data['influence_f_md']  - data['influence_f_md_min'])   / (data_norm['influence_f_md'] + data_norm['influence_e_md'])
    clustr_data['influence_e_md'] = (clustr_data['influence_e_md']  - data['influence_e_md_min'])   / (data_norm['influence_e_md'] + data_norm['influence_f_md'])

    # return cluster and associated data
    return planet_data, clustr_data

# --- game architecture function ---
def compute_orders(turn, pid, planets, fleets, train, gships):
    '''

    '''
    t0 = time.time()
    # hard coded decay & cluster radius
    d = 1.0 
    cluster_outer_radius = 4 # int(t_max * 30 / 100) 

    # compute distances between all planets and get t_max
    distances, t_max = compute_dist_and_t_max(planets)
    # add fleets in dictionary format for fast access in Value function
    fleets_dict, my_planets, fleets_from_this_planet = compute_static_data(pid, planets, t_max)    

    for fleet in fleets:
        fleets_dict[fleet.owner][fleet.destination][int(fleet.remaining_turns)] += fleet.ships
        fleets_dict[0][fleet.destination][int(fleet.remaining_turns)] += fleet.ships
        fleets_from_this_planet[fleet.source] = True

    # compute overall growth and ship ratios
    game_metrics = compute_game_growth_ships_ratios(pid, turn, planets, fleets)
    if game_metrics == None: return []
    
    if len(my_planets) == 0: 
        from planetwars.neuralnetwork import NNetwork       
        NNetwork.train(turn, pid, game_metrics, None, True)
        return []

    # compute initial data    
    data, data_norm = compute_data(pid,
                                   planets,
                                   fleets_dict,
                                   distances, 
                                   t_max, 
                                   d, 
                                   cluster_outer_radius,
                                   game_metrics)   
    
    my_planets = [p for p in my_planets if (p.ships > 20 and data['value'][p.id][t_max] > 0.0)] 
        
    # my_planets = [p for p in my_planets if fleets_from_this_planet[p.id] == False]
    if len(my_planets) == 0: return []
    p_clusters = associate_clusters(planets, 
                                    my_planets,
                                    distances,                                            
                                    cluster_outer_radius)

    # main loop. We do this until we have no more relevant planets
    # we declare the array that will contain order for this turn!
    orders = []
    new_orders = []
    for cls in sorted(p_clusters, key=lambda tup: tup[2]): # sort by distance:
         
        # copy data from tuple to local easy access variables
        planet, cluster, t_dist = cls[0], cls[1], cls[2]
        if len(my_planets) == 0: break;

        # filter cluster, maybe some planets were used in previous cluster and should do nothing this turn
        cluster = [p for p in cluster if p in my_planets]        
        if planet == None or len(cluster) == 0: continue # we do nothing this turn

        # get cluster and associated data
        p_data, cluster_data = compute_cluster_data(planet, 
                                                    cluster, 
                                                    t_dist,
                                                    distances,   
                                                    d,                                                                                                                                  
                                                    cluster_outer_radius,
                                                    data,
                                                    data_norm) 
        
        # create orders if possible / valid
        # we can have multiple orders here, because we have a cluster of planets
        new_orders.append(gships(pid, 
                            p_data, 
                            cluster_data, 
                            t_dist,
                            t_max,
                            data,
                            turn,
                            train, 
                            planets, 
                            fleets,
                            game_metrics))  

    sorted_orders = sorted(new_orders, key=lambda tup: tup[0], reverse=True)
    if train: # random action selection
        global random_move
        if random.uniform(0,1) < random_move: 
            if random.uniform(0,1) > 0.3: sorted_orders = sorted(new_orders, key=lambda tup: tup[3], reverse=False)
            else: sorted_orders = sorted(new_orders, key=lambda tup: tup[4], reverse=False)
        random_move -= 0.00001

    if len(sorted_orders) == 0: return []
    # if sorted_orders[0][0] <= 0.5: 
    orders += sorted_orders[0][1]
        
    if train:
        # train saved patterns
        from planetwars.neuralnetwork import NNetwork
        # first train from previous saved patterns, then add the new one        
        NNetwork.train(turn, pid, game_metrics, [s[2]['in'] for s in sorted_orders])
        NNetwork.save_pattern(pid, sorted_orders[0][2])

    t1 = time.time()
    # print ('avg computing time: ', t1 - t0, ' seconds')    
    return orders
