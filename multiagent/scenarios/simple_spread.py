import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# np.random.seed(101)
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 4
        world.collaborative = False

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.2
            if i < 2:
                agent.color = np.array([0.9, 0.2, 0.5])
                agent.team = 'A'
            else:
                agent.color = np.array([0.35, 0.35, 0.85])
                agent.team = 'B'

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
        # for i, agent in enumerate(world.agents):
        #     agent.color = np.array([0.35, 0.35, 0.85])
        #     if i >= 2:
        #         agent.color = np.array([0.9, 0.2, 0.5])
        # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     if i == 0:
        #         landmark.color = np.array([1, 0, 0])
        #     if i == 1:
        #         landmark.color = np.array([0, 1, 0])
        #     if i == 2:
        #         landmark.color = np.array([0,0, 1])
        #     if i == 3:
        #         landmark.color = np.array([1, 1, 0])
        # 
        # set random initial states

        for i, agent in enumerate(world.agents):


            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # print(agent.state.p_pos)
            # if i == 0:
            #     agent.state.p_pos = np.array([0.7, +0.8])
            # if i == 1:
            #     agent.state.p_pos = np.array([0.7, +0.4])
            # if i == 2:
            #     agent.state.p_pos = np.array([-0.6, -0.4])
            # if i == 3:
            #     agent.state.p_pos = np.array([-0.6, -0.8])

            # a = np.array([0, 1, 2, 3])
            # np.random.shuffle(a)
            # print(a)

            # if i == 0:
            #     agent.state.p_pos = np.array([+0.6, +0.6])
            # if i == 1:
            #     agent.state.p_pos = np.array([-0.6, +0.6])
            # if i == 2:
            #     agent.state.p_pos = np.array([-0.6, -0.6])
            # if i == 3:
            #     agent.state.p_pos = np.array([+0.6, -0.6])

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            #
            # if i == 0:
            #     landmark.state.p_pos = np.array([0.0, +0.8])
            # if i == 1:
            #     landmark.state.p_pos = np.array([0.0, -0.3])
            # if i == 2:
            #     landmark.state.p_pos = np.array([0.0, +0.3])
            # if i == 3:
            #     landmark.state.p_pos = np.array([0.0, -0.8])


            if i == 0:
                landmark.state.p_pos = np.array([0.0, -0.6])
            if i == 1:
                landmark.state.p_pos = np.array([+0.6, +0.0])
            if i == 2:
                landmark.state.p_pos = np.array([-0.6, +0.0])
            if i == 3:
                landmark.state.p_pos = np.array([0.0, +0.6])



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
        # for l in world.landmarks:
        #
        #     for a in world.agents
        #         dists.append(np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))))
        #
            # dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            # rew -= min(dists)

        dists = [np.sum(np.square(agent.state.p_pos - l.state.p_pos)) for l in world.landmarks]
        # print(dists)
        rew = - min(np.sqrt(dists))

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

        # if agent.team == 'A':
        #     rew = 0.0
        # elif agent.team == 'B':
        #     rew = 1.0
        # else:
        #     rew = 0.0
        #     print(agent.team)

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
        # print('Another agent')
        for i, other in enumerate(world.agents):
            # print(str(i) + str(other.team))

            if other is agent: continue
            comm.append(other.state.c)
            p_noise = np.array([0, 0])

            if other.team == agent.team and agent.team == 'A':
                p_noise = np.random.normal(0,0.0, size=2)
            elif other.team == agent.team and agent.team == 'B':
                p_noise = np.random.normal(0,0.2, size=2)
            else:
                p_noise = np.random.normal(0,1.0, size=2)



            # if agent.team == 'A':
            #     # p_noise = np.random.normal(0,0.0, size=2)
            #     p_noise = np.array([0, 0])
            # elif agent.team == 'B':
            #     # p_noise = np.random.normal(0,10.0, size=2)
            #     p_noise = -(other.state.p_pos - agent.state.p_pos)




            # print(other.state.p_pos)
            # print(p_noise)
            # print('\n')
            other_pos.append(other.state.p_pos + p_noise - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
