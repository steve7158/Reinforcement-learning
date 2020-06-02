import gym
import numpy as np
from tqdm import tqdm

env=gym.make('CartPole-v1')
LEARNING_RATE=0.1
DISCOUNT=0.95
EPISODES=30000
SHOW_EVERY=3000
DISCRETE_OS_SIZE=[20]*len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

epsilon=1
START_EPSILON_DECAYING=1
END_EPSILON_DECAYING=EPISODES//2
epsilon_decay_value=epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

q_table=np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE+[env.action_space.n]))

def get_discrete_state(state):
    discrete_state=(state-env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

episode_division=EPISODES//SHOW_EVERY
for i in range(episode_division):
    for episode in tqdm(range(SHOW_EVERY)):
        discrete_state=get_discrete_state(env.reset())
        done=False

        if episode==SHOW_EVERY-1:
            render=True
            print(i)
        else:
            render=False
        while not done:
            if np.random.random()>epsilon:
                action=np.argmax(q_table[discrete_state])
            else:
                action=np.random.randint(0, env.action_space.n)
            new_state, reward, done, _=env.step(action)
            new_discrete_state=get_discrete_state(new_state)

            if render:
                env.render()
            if not done:
                max_future_q=np.max(q_table[new_discrete_state])
                current_q=q_table[discrete_state+(action,)]
                new_q=(1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_future_q)
                q_table[discrete_state+(action,)]=new_q
            # elif new_state[0] >= env.goal:
            #     q_table[discrete_state+(action,)]=0
            discrete_state=new_discrete_state
        if END_EPSILON_DECAYING>=episode>=START_EPSILON_DECAYING:
            epsilon-=epsilon_decay_value
print(q_table[-1])
while True:
    env.render()
    
env.close()
