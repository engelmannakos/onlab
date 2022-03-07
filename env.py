import gym
from gym import spaces, utils
import traci
import sys, os
import numpy as np
import sumolib
import random
import math
sys.path.append(os.path.join(os.environ["SUMO_HOME"],"tools"))

class NaviEnv(gym.Env):

	"""
		A majdnem összes self. változó:
			self.action_space # Az action space
			self.observation_space # Az observation space
			self.steps # Az agent által megtett lépések egy epizód alatt
			self.sumo_steps # A sumo által megtett lépések egy epizód alatt
			self.car # Az agent vehicleID-ja
			self.done # Véget ért-e az epizód
			self.reward # Jelenlegi reward az epizódban
			self.prev_lanes # Korábban bejárt lane-ek listája
			self.last_loopId # A legutóbbi loop ID-ját tárolja, ugyanis egy loop kb. 3 lépésen keresztül
							 # érzékeli az autót, de nekünk elég ha az elsőnél dönt
			self.lanes # A pálya összes lane-je
			self.target # A cél lane ID-ja
			selg.target_list # A lehetséges célok listája
	"""
	edge_num = 73
	target_edge = "gneE59"
	junctions_num = 41.0
	avenue = ["-gneE58", "-gneE52", "-gneE53", "-gneE54", "-gneE55", "gneE6", "gneE39", "gneE59"] # A legforgalmasabb útvonal szakaszai
	Cars = True
	

	def __init__(self):
		self.net = sumolib.net.readNet('f:/NeteditNetworks/manhattan/manhattan.net.xml')
		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Dict({
			'position': spaces.MultiDiscrete([41,41]),
			'time': spaces.Discrete(4000),
			'density': spaces.Box(low=0.0, high=15.0, shape=(73,), dtype=np.float32)
		})
		
		self.steps = 0
		self.target_list = ["gneE58"]
		self.time_diff = 0


	def step(self, action): # Amint döntöttünk egy action mellett, ez a függvény lefut
		self.steps += 1
		self.reward = 0

		if self.Cars:
			if len(self.car_stop_ids) != 3: # Mindig max 3 autót állítok meg
				idx = random.randint(0, traci.vehicle.getIDCount()-1)
				
				# Ezek az autók nem lehetnek olyan útszakaszon, aminek csak 1 sávja van, mert ha mögötte áll az agent, akkor sosem ér el az érzékelőig
				while traci.vehicle.getIDList()[idx] == self.car or traci.vehicle.getIDList()[idx] in self.car_stop_ids or traci.edge.getLaneNumber(traci.lane.getEdgeID(traci.vehicle.getLaneID(traci.vehicle.getIDList()[idx]))) < 2:
					idx = random.randint(0, traci.vehicle.getIDCount()-1)
				car_id = traci.vehicle.getIDList()[idx]
				self.car_stop_ids.append([car_id, 0])
				traci.vehicle.setSpeed(car_id, 0)
				traci.vehicle.setColor(car_id, (0,0,255)) # Csak hogy könnyebben lássam a térképen, mely autókról van szó

			if len(self.car_stop_ids) > 0:
				for cars in self.car_stop_ids:
					if cars[1] == 4:
						try:
							traci.vehicle.setSpeed(cars[0], -1)
							traci.vehicle.setColor(cars[0], (255,255,0))
							self.car_stop_ids.remove(cars)
						except Exception as e:
							# Itt ha jól emlékszem olyan probléma volt, hogy a rendszer egy idő után az álló kocsikat kivette,
							# de ezt az időt később elég nagyra állítottam, hogy ilyen ne legyen. Nem tudom viszont, hogy csak ez volt-e
							# a probléma, úgyhogy benne hagytam ezt
							print("Valahova eltűnt az autó, mi a fene, Exception:", e)
							self.car_stop_ids.remove(cars)
					else:
						cars[1] += 1
		
		linknum = 0

		# Megkeresem a lane-t, amin állunk.
		for l in self.lanes:
			if self.current_edge == l.split('_')[0]: # Jobb megoldás híján (a sávok (lane) a szakaszok (edge) után kapják a nevüket, pl gneE01 az edge, akkor gneE01_0 és gneE01_1 a két lane)
				num = traci.lane.getLinkNumber(l)
				if num > linknum:
					linknum = num
					links = traci.lane.getLinks(l) # Ugyan a legnagyobbat kapja meg, ezáltal pl egy: ^|^> útnál a 0 action minden esetben jobbrát jelent, a bal lane-en is

		if self.current_edge in self.prev_edges: # Már jártunk ezen az edge-n
			traci.close()
			return self.observation_space, -1, True, {} # Visszaadom az állapotteret, a rewardot, és a boolt, hogy véget ért-e az epizód
		else:
			self.prev_edges.append(self.current_edge)

		if (action+1) > linknum:
			action = 0
			# Ezt említettem élőben, itt eredetileg az volt, hogy ha nem létező útra
			# akar menni, akkor kezdjen új epizódot, ez változott később arra,
			# hogy a 0-s irányba menjen inkább

			# traci.close()
			#return self.observation_space, -1, True, {}
		
		next_edge = ""

		if action == 0:
			next_edge = traci.lane.getEdgeID(links[0][0])
			traci.vehicle.setRoute(self.car, [self.current_edge, next_edge])
			print("Next", next_edge)

		if action == 1:
			next_edge = traci.lane.getEdgeID(links[1][0])
			traci.vehicle.setRoute(self.car, [self.current_edge, next_edge])
			print("Next", next_edge)


		if action == 2:
			next_edge = traci.lane.getEdgeID(links[2][0])
			traci.vehicle.setRoute(self.car, [self.current_edge, next_edge])
			print("Next", next_edge)

		self.traci_step() # Fut a szimuláció

		self.current_lane = traci.vehicle.getLaneID(self.car)
		self.current_edge = traci.lane.getEdgeID(self.current_lane)

		self.observation_space = self.fill_obs() # Feltöltöm az állapotteret

		### Ez már csak egy utolsó pillanatban tett próbálkozás, nem lett átgondolva
		pos = traci.vehicle.getPosition(self.car)
		dist = math.sqrt((pos[0]-self.endpoint[0])**2 + (pos[1]-self.endpoint[1])**2) 

		if dist < self.distance:
			self.distance = dist
			self.reward += 0.1
		else:
			self.reward -= 0.1
		print("Dist:", self.distance)
		###

		if self.done:
			traci.close()

		return self.observation_space, self.reward, self.done, {}

	def reset(self):

		if self.Cars:

			# Sűrű forgalom az avenue-n, hogy feltöltődjön forgalommal az agent előtt
			os.system('python f:/NeteditNetworks/navi_grid/trips/randomTrips.py '+
				'-n f:/NeteditNetworks/manhattan/avenue.net.xml '+
				'-e 50 '+
				'--prefix av_ '+
				'--route-file f:/NeteditNetworks/manhattan/avenue_trip.rou.xml '+
				'-p 0.08')

			# Átlagos forgalom az agent megjelenésével kb egyidőben
			os.system('python f:/NeteditNetworks/navi_grid/trips/randomTrips.py '+
				'-n f:/NeteditNetworks/manhattan/avenue.net.xml '+
				'-e 2000 '+
				'--fringe-factor 10000 '+
				'-b 230 '+
				'--prefix av2_ '+
				'--route-file f:/NeteditNetworks/manhattan/avenue_trip2.rou.xml '+
				'-p 0.3')
			
			os.system('sumo --save-configuration f:/NeteditNetworks/manhattan/manhattan.sumocfg '+
				'--net-file f:/NeteditNetworks/manhattan/manhattan.net.xml '+
				'--route-files f:/NeteditNetworks/manhattan/avenue_trip.rou.xml,f:/NeteditNetworks/manhattan/avenue_trip2.rou.xml,f:/NeteditNetworks/manhattan/base_trip.rou.xml '+
				'--time-to-teleport 30000 '+
				'--additional-files f:/NeteditNetworks/manhattan/detectors.add.xml')
		
		
		self.target_junc = self.net.getEdge(self.target_edge).getToNode().getID()
		self.steps = 0
		self.sumo_steps = 0
		self.done = False
		self.reward = 0
		self.last_loopId = ""
		self.prev_edges = []
		self.time_diff = 0
		self.car_stop_ids = []
		self.distance = 0
		self.endpoint = (1905.05,897.36)
		if self.Cars:
			traci.start(["sumo", "-c", "f:/NeteditNetworks/manhattan/manhattan.sumocfg", "--step-length", "1"])
		else:
			traci.start(["sumo", "-c", "f:/NeteditNetworks/manhattan/proba.sumocfg", "--step-length", "1"])
		traci.simulationStep()
		self.edges = []
		self.junctions = []
		for edg in traci.edge.getIDList(): # Ez amolyan 'működik, úgyhogy nem babrálok vele' - megoldás
			if 'gneE' in edg:
				self.edges.append(edg)
		
		self.junctions = traci.junction.getIDList()[5:]
		self.lanes = []
		for l in traci.lane.getIDList():
			if "E" in l:
				self.lanes.append(l)
		self.car = 'agent_trip' # Megintcsak az egyszerűség kedvéért nem foglalkoztam vele később, hogy normálisan legyen megírva
		while self.car not in traci.vehicle.getIDList():
			traci.simulationStep()
		self.time_start = traci.simulation.getTime()
		self.current_lane = traci.vehicle.getLaneID(self.car)
		self.current_edge = traci.lane.getEdgeID(self.current_lane)

		traci.vehicle.setRoute(self.car, [self.current_edge])

		self.traci_step()
		self.observation_space = self.fill_obs()

		###
		pos = traci.vehicle.getPosition(self.car)
		self.distance = math.sqrt((pos[0]-self.endpoint[0])**2 + (pos[1]-self.endpoint[1])**2) 
		###

		return self.observation_space

	def render(self):
		...

	def fill_obs(self):
		obs_data = []
		self.current_lane = traci.vehicle.getLaneID(self.car)
		self.current_edge = traci.lane.getEdgeID(self.current_lane)
		self.current_junc = self.current_edge

		x = self.junctions.index(self.current_junc)
		y = self.junctions.index(self.target_junc)

		for e_id in self.edges:
			l_id = e_id + "_0" # ismét egy 'így jó lesz, majd megírom később' kódrészlet
			lane_num = traci.edge.getLaneNumber(e_id)
			data = traci.edge.getLastStepVehicleNumber(e_id) / (traci.lane.getLength(l_id)*lane_num)
			obs_data.append(data)

		self.time_end = traci.simulation.getTime()
		self.time_diff = self.time_end - self.time_start
		observ = {
			'position': [x,y],
			'time': self.time_diff,
			'density': obs_data
		}
		return observ

	def traci_step(self):
		next_step = False
		while next_step == False:
			traci.simulationStep()
			self.time_end = traci.simulation.getTime()
			self.time_diff = self.time_end - self.time_start
			
			# Ezt amiatt tettem ide, mivel ennyi idő alatt ha nem jutott el a célba,
			# akkor már biztos rossz útvonalon megy, és mivel 1-2 napig futott
			# egy tanítás, kellettek ilyen megkötések
			if self.time_diff > 3000:
				next_step=True
				self.done=True
				self.reward=-1
			self.sumo_steps += 1
			loopID_List = traci.inductionloop.getIDList()
			for loopID in loopID_List:
				if traci.inductionloop.getLastStepVehicleNumber(loopID) > 0:
					loopCar_List = traci.inductionloop.getLastStepVehicleIDs(loopID)
					for loopCar in loopCar_List:
						if self.car == loopCar and self.last_loopId != loopID:
							self.last_loopId = loopID
							next_step = True
							if traci.lane.getEdgeID(traci.inductionloop.getLaneID(loopID)) == self.target_edge: # Kijutottunk
								self.time_end = traci.simulation.getTime()
								self.time_diff = self.time_end - self.time_start
								self.reward = 8 - self.time_diff/100
								if self.reward < 1 :
									self.reward = 1
								self.done = True

							# A nevezékről annyit, hogy eredetileg random kijárathoz kellett volna elérnie, ezért lett target_list
							# Ha rossz kijárathoz érkeztünk, akkor megintcsak vége
							elif traci.lane.getEdgeID(traci.inductionloop.getLaneID(loopID)) in self.target_list:
								self.reward = -1
								self.done = True


