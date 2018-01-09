The AI module at UC Berkeley has nice resouces in Python including a version of Pacman for exploring reinforcement learning. Homepage: http://ai.berkeley.edu/reinforcement.html The Python resources can be downloaded from: https://s3-us-west-2.amazonaws.com/cs188websitecontent/projects/release/ reinforcement/v1/001/reinforcement.zip

The mLearningAgents.py file implements QLearningAgent for Pacman.

Replace this file in the reinforcement folder.

Run the code by using the following command:

python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid

-p QLearnAgent tells pacman.py code to let Pacman take controlled by an object that is an instance of a class QLearnAgent.

-x 2000 will train the learner for 2000 episodes (you can use any number)

-n 2010 will run it for 10 non-training episodes

-l smallGrid runs the very reduced game
