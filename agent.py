import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd




class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions=[None, 'forward', 'left', 'right']
        #data and erros are for storing metrics
        self.data = pd.DataFrame(columns=['total_reward', 'total_hits', 'total_errors', 'total_nav_error', 'total_redcrash', 'time','deadline','deadline_remaining'])
        self.errors = pd.DataFrame(columns=['iteration', 'input', 'waypoint', 'exploring', 'action'])
        self.route = pd.DataFrame(columns=['iteration', 'time','position','heading', 'destination','action','waypoint','input'])
        self.exploring=False
        self.error_count=0
        self.initQ()
        self.state = ('init')
        self.count=0
        self.current_iteration=0
        self.prev_data = {'state': 'init', 'action': None, 'reward': 0}
        self.alpha = 0.8
        self.gamma = 0.4
        self.epsilon = 0.5
        self.epsilon_decay=0.0005

    def initQ(self):
        #initialize all the possible keys of Q
        self.Q = {}
        self.Q['init'] = 0.0
        for light in ['red','green']:
            for oncoming in [None, 'forward', 'left', 'right']:
                for left in [None, 'forward', 'left', 'right']:
                    for right in [None, 'forward', 'left', 'right']:
                        for waypoint in ['forward', 'left', 'right']:
                                for action in self.actions:
                                    self.Q[(light,oncoming,left,right,waypoint,action)]= 0.0


    def reset(self, destination=None):
        self.current_iteration+=1
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.destination=destination
        self.state = ('init')
        self.count = 0
        self.prev_data = {'state': 'init', 'action': None, 'reward': 0}
        self.data.loc[self.current_iteration]=[0.0 for i in range(len(self.data.columns))]
        self.route.loc[len(self.route)] = [self.current_iteration,self.count,self.env.agent_states[self]['location'],self.env.agent_states[self]['heading'],self.destination,None,None,None]


    def chooseAction(self, state):
        #Choose Action. If is the first time use random chance. Also check if random<epsilon.
        self.exploring=True
        if (self.count==0 and self.current_iteration==1) :
            return random.choice(self.actions)
        if random.random()<self.epsilon:
            return random.choice(self.actions)
        self.exploring=False
        q = [self.Q[state+(a,)] for a in self.actions]
        all_best=[i for i, x in enumerate(q) if x == max(q)]
        return self.actions[random.choice(all_best)]

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        position= self.env.agent_states[self]['location']
        heading = self.env.agent_states[self]['heading']
        # TODO: Update state
        self.state = (inputs['light'],inputs['oncoming'],inputs['left'],inputs['right'],self.next_waypoint)
        # TODO: Select action according to your policy

        action = self.chooseAction(self.state)
        #Update epsilon
        self.epsilon=max(self.epsilon-self.epsilon_decay,0)
        # Execute action and get reward
        reward = self.env.act(self, action)
        # TODO: Learn policy based on state, action, reward
        if self.count != 0:
            self.Q[self.prev_data['state']+(self.prev_data['action'],)]=self.Q[self.prev_data['state']+(self.prev_data['action'],)]*(1-self.alpha)+ \
                                                                        self.alpha *(self.prev_data['reward']+self.gamma*max([self.Q[self.state+(a,)] for a in self.actions]))


        #store current values to use in next iteration
        self.prev_data = {'state' : self.state, 'action' : action, 'reward' : reward}

        #store metrics in panda
        self.count+=1
        self.route.loc[len(self.route)] = [self.current_iteration, self.count, position,heading,self.destination,action,self.next_waypoint,inputs]
        self.data['total_reward'][self.current_iteration]+=reward
        self.data['total_hits'][self.current_iteration] += 1 if reward>0.0 else 0
        self.data['total_errors'][self.current_iteration] += 1 if reward<0.0 else 0
        self.data['total_nav_error'][self.current_iteration] += 1 if reward == -0.5 else 0
        self.data['total_redcrash'][self.current_iteration] += 1 if reward <= -1.0 else 0
        self.data['time'][self.current_iteration] = self.count-1
        self.data['deadline_remaining'][self.current_iteration] = deadline
        if self.count==1:
            self.data['deadline'][self.current_iteration] = deadline

        if reward<0.0:
            self.errors.loc[self.error_count] = [self.current_iteration,inputs,self.next_waypoint,self.exploring,action]
            self.error_count+=1

        #print "Debug: destination = {}, waypoint = {}".format(self.destination,self.next_waypoint)
        #print "State: destination = {}".format(self.state)
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def end(self):
        #Implemented this method by modifying simulator so it calls this method when simulation ends.
        #Used to save all metrics in cvs
        self.data.to_csv("model_metrics.csv", sep=';',decimal=",")
        self.errors.to_csv("errors_log.csv", sep=';',decimal=",")
        self.route.to_csv("route.csv", sep=';', decimal=",")
        storeQ=pd.DataFrame(columns=['key', 'q'])
        i=0
        for k,q in self.Q.items():
            storeQ.loc[i]=[k,q]
            i+=1
        storeQ.to_csv("Q.csv", sep=';',decimal=",")






def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

