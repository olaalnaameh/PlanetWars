import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ai 
import json
import random
import string
import time
from flask import abort, Flask, redirect, render_template, Response, request
from gevent import monkey
from random import choice
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from socketio.server import SocketIOServer
from threading import Thread
from werkzeug.serving import run_with_reloader

from planetwars import PlanetWars
from planetwars.internal import natural_key
from planetwars.views import RealtimeView
from ai.state import State

app = Flask(__name__)
monkey.patch_all()

games = {}

class GamesNamespace(BaseNamespace):

    games = {}

    def on_join(self, game):
        self.game = game
        GamesNamespace.games[self.game].client_join(self)

    def disconnect(self, *args, **kwargs):
        if hasattr(self, "game") and self.game in GamesNamespace.games:
            del GamesNamespace.games[self.game].sockets[id(self)]
        super(GamesNamespace, self).disconnect(*args, **kwargs)

    @classmethod
    def emit_to_game(self, game, event, *args):
        for ns in self.games[game].sockets.itervalues():
            ns.emit(event, *args)

class WebsocketView:

    def __init__(self, game):
        self.game = game
        self.sockets = {}
        GamesNamespace.games[game] = self

    def client_join(self, client):
        self.sockets[id(client)] = client
        client.emit("initialize", json.dumps({
            "turnsPerSecond": self.turns_per_second,
            "planets": self.planets,
            "map": self.map_name,
            "players": self.players,
        }))

    def initialize(self, turns_per_second, planets, map_name, players):
        self.turns_per_second = turns_per_second
        self.planets = [p.freeze()._asdict() for p in planets]
        self.map_name = map_name
        self.players = players

    def update(self, planets, fleets):
        asdicts = [p._asdict() for p in planets], [f._asdict() for f in fleets]
        GamesNamespace.emit_to_game(self.game, "update", json.dumps(asdicts))

    def game_over(self, winner, ship_counts, turns):
        GamesNamespace.emit_to_game(self.game, "gameOver", json.dumps({
          "winner": winner,
          "shipCounts": ship_counts,
          "turns": turns
        }))
        del GamesNamespace.games[self.game]
        if self.game in games:
            del games[self.game]

@app.route("/")
def index():
    return render_template("index.html",
            ai_names=sorted(PlanetWars.ais.keys(), key=natural_key),
            map_names=sorted(PlanetWars.maps.keys(), key=natural_key))

@app.route("/game/<path:game_id>")
def game(game_id):
    if game_id in games:
        return render_template("game.html", gameID=game_id)
    else:
        return redirect("/")

@app.route("/create-game", methods=["POST"])
def create_game():
    game_id = "".join(choice(string.lowercase) for _ in range(5))
    p1 = request.form["p1"]
    p2 = request.form["p2"]
    m = request.form.get("map", "Random")
    planets = None
    fleets = None
    state = State()

    if m == "Random":
        m = random.choice(PlanetWars.maps.keys())
    if m == "Generate":
        m = None
        state.random_setup(2, 2, 10)
    c = request.form["collisions"]
    if c == "True":
        c = True
    else:
        c = False
                        
    tps = float(request.form.get("tps", 8))
    seed = random.randint(1,2000000000)
    print ("############ SEED=", seed)
    
    games[game_id] = PlanetWars([p1, p2], m, state.planets, state.fleets, turns_per_second=tps, collisions=c)
    view = WebsocketView(game_id)
    games[game_id].add_view(view)
    Thread(target=games[game_id].play).start()
    return redirect("/game/" + game_id)

@app.route("/socket.io/<path:rest>")
def push_stream(rest):
    try:
        socketio_manage(request.environ, {"/game": GamesNamespace}, request)
    except:
        app.logger.error("Exception while handling socketio connection",
                         exc_info=True)
    return Response()

# @run_with_reloader
def run_dev_server():
    app.debug = True
    port = 4200
    print (" * Running on http://127.0.0.1:%i/" % port)
    SocketIOServer(("", port), app, resource="socket.io").serve_forever()

if __name__ == "__main__":
    run_dev_server()
