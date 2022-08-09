import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# np.random.seed(101)
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 6
        num_landmarks = 6
        world.collaborative = False

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % (i+1)
            agent.collide = True
            agent.silent = True
            agent.size = 0.17
            # if i < 2:
            #     agent.color = np.array([0.9, 0.2, 0.5])
            #     agent.team = 'A'
            # else:
            #     agent.color = np.array([0.35, 0.35, 0.85])
            #     agent.team = 'B'

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        BLUE = [0.12, 0.46, 0.70]
        for i, agent in enumerate(world.agents):
            agent.color = np.array(BLUE) ## BLUE
            # if agent.name == 'agent 1':
            #     agent.color = np.array([0.84, 0.15, 0.15]) ## BLUE
            # if agent.name == 'agent 2':

            # if agent.name == 'agent 3':
            #     agent.color = np.array(BLUE) ## BLUE
            # if agent.name == 'agent 4':
            #     agent.color = np.array(BLUE) ## BLUE
            # if agent.name == 'agent 5':
            #     agent.color = np.array(BLUE) ## BLUE
            # if agent.name == 'agent 6':
            #     agent.color = np.array(BLUE) ## BLUE


            # if agent.name == 'agent 1':
            #     agent.color = np.array([0.12, 0.46, 0.70]) ## BLUE
            # if agent.name == 'agent 2':
            #     agent.color = np.array([0.12, 0.46, 0.70]) ## BLUE
            # if agent.name == 'agent 3':
            #     agent.color = np.array([0.12, 0.46, 0.70]) ## BLUE
            # if agent.name == 'agent 4':
            #     agent.color = np.array([0.84, 0.15, 0.15]) ## RED




        # random properties for landmarks
        RED = [0.84, 0.15, 0.15]
        for i, landmark in enumerate(world.landmarks):
            # if i == 0:
            landmark.color = np.array(RED)
            # if i == 1:
            #     landmark.color = np.array(RED)
            # if i == 2:
            #     landmark.color = np.array(RED)
            # if i == 3:
            #     landmark.color = np.array(RED)
            # if i == 4:
            #     landmark.color = np.array(RED)
            # if i == 5:
            #     landmark.color = np.array(RED)
            # if i == 6:
            #     landmark.color = np.array(RED)
        #
        # set random initial states

        for i, agent in enumerate(world.agents):


            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)


            # if agent.name == 'agent 1':
            #     agent.state.p_pos = np.array([+0.6, +0.6])
            # if agent.name == 'agent 2':
            #     agent.state.p_pos = np.array([-0.6, +0.6])
            # if agent.name == 'agent 3':
            #     agent.state.p_pos = np.array([-0.6, -0.6])
            # if agent.name == 'agent 4':
            #     agent.state.p_pos = np.array([+0.6, -0.6])
            #




            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

            # if i == 0:
            #     landmark.state.p_pos = np.array([0.0, -0.6])
            # if i == 1:
            #     landmark.state.p_pos = np.array([+0.6, +0.0])
            # if i == 2:
            #     landmark.state.p_pos = np.array([-0.6, +0.0])
            # if i == 3:
            #     landmark.state.p_pos = np.array([0.0, +0.6])
            r = 0.7
            landmark.state.p_pos = np.array([r*np.cos((i)*(2*np.pi)/6), r*np.sin((i)*(2*np.pi)/6)])





    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        ### Eyes
        # if agent.name == 'agent 1':
        #     network = np.array([1, 0, 0, 0, 0, 0])
        # if agent.name == 'agent 2':
        #     network = np.array([0, 1, 0, 0, 0, 0])
        # if agent.name == 'agent 3':
        #     network = np.array([0, 0, 1, 0, 0, 0])
        # if agent.name == 'agent 4':
        #     network = np.array([0, 0, 0, 1, 0, 0])
        # if agent.name == 'agent 5':
        #     network = np.array([0, 0, 0, 0, 1, 0])
        # if agent.name == 'agent 6':
        #     network = np.array([0, 0, 0, 0, 0, 1])



        ### Ones
        if agent.name == 'agent 1':
            network = np.array([1, 1, 1, 1, 1, 1])
        if agent.name == 'agent 2':
            network = np.array([1, 1, 1, 1, 1, 1])
        if agent.name == 'agent 3':
            network = np.array([1, 1, 1, 1, 1, 1])
        if agent.name == 'agent 4':
            network = np.array([1, 1, 1, 1, 1, 1])
        if agent.name == 'agent 5':
            network = np.array([1, 1, 1, 1, 1, 1])
        if agent.name == 'agent 6':
            network = np.array([1, 1, 1, 1, 1, 1])


        ### Star
        # if agent.name == 'agent 1':
        #     network = np.array([1, 1, 1, 1, 1, 1])
        # if agent.name == 'agent 2':
        #     network = np.array([1, 1, 0, 0, 0, 0])
        # if agent.name == 'agent 3':
        #     network = np.array([1, 0, 1, 0, 0, 0])
        # if agent.name == 'agent 4':
        #     network = np.array([1, 0, 0, 1, 0, 0])
        # if agent.name == 'agent 5':
        #     network = np.array([1, 0, 0, 0, 1, 0])
        # if agent.name == 'agent 6':
        #     network = np.array([1, 0, 0, 0, 0, 1])



        ### Lattice
        # if agent.name == 'agent 1':
        #     network = np.array([1, 1, 0, 0, 0, 1])
        # if agent.name == 'agent 2':
        #     network = np.array([1, 1, 1, 0, 0, 0])
        # if agent.name == 'agent 3':
        #     network = np.array([0, 1, 1, 1, 0, 0])
        # if agent.name == 'agent 4':
        #     network = np.array([0, 0, 1, 1, 1, 0])
        # if agent.name == 'agent 5':
        #     network = np.array([0, 0, 0, 1, 1, 1])
        # if agent.name == 'agent 6':
        #     network = np.array([1, 0, 0, 0, 1, 1])




        ### Centeralized Reward
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     # dists = dists * network
        #     rew -= min(dists)
        #     # rew -= np.dot(dists,network)
        #     # rew -= np.dot(dists,network)/np.sum(network)



        ### Decenteralized Reward
        # dists = [np.sum(np.square(agent.state.p_pos - l.state.p_pos)) for l in world.landmarks]
        # rew = - min(np.sqrt(dists))

        ### Networked Reward
        personal_rew = []
        if 'network' in locals():
            for a in world.agents:
                dists = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) for l in world.landmarks]
                personal_rew.append(-min(np.sqrt(dists)))
            if np.sum(network) != 0:
                rew = np.dot(personal_rew,network)/np.sum(network)


        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []

        for i, other in enumerate(world.agents):


            if other is agent: continue
            comm.append(other.state.c)




            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # np.random.shuffle(entity_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
