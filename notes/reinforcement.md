# Reinforcement Learning

###### tags: `FreeCodeCamp-MLwithPython`


* [Terminology](#terminology)
    * [Environment](#environment)
    * [Agent](#agent)
    * [State](#state)
    * [Action](#action)
    * [Reward](#reward)
* [Q-Learning](#q-learning)
    * [Learning the Q-Table](#learning-the-q-table)
    * [Updating Q-Values](#updating-q-values)
    * [Learning Rate α](#learning-rate-α)
    * [Discount Factor γ](#discount-factor-γ)
* [Sources](#sources)


---

This technique is different than many of the other machine learning techniques we have seen earlier and has many applications in training agents (an AI) to interact with enviornments like games. Rather than feeding our machine learning model millions of examples we let our model come up with its own examples by **exploring an enviornment**. Humans learn by exploring and learning from mistakes and past experiences so let's have our computer do the same.

The most important part of reinforcement learning is determing how to reward the agent. After all, the goal of the agent is to **maximize its rewards**. This means we should reward the agent appropiatly such that it reaches the desired goal.

---

## Terminology

### Environment

Iin reinforcement learning tasks we have a notion of the enviornment. This is what our agent will explore. An example of an enviornment in the case of training an AI to play say a game of mario would be the level we are training the agent on.

### Agent

An agent is an entity that is exploring the enviornment. Our agent will interact and take different actions within the enviornment. In our mario example the mario character within the game would be our agent. 

### State

Always our agent will be in what we call a state. The state simply tells us about the status of the agent. The most common example of a state is the location of the agent within the enviornment. Moving locations would change the agents state.

### Action

Any interaction between the agent and enviornment would be considered an action. For example, moving to the left or jumping would be an action. An action may or may not change the current state of the agent. In fact, the act of doing nothing is an action as well! The action of say not pressing a key if we are using our mario example.

### Reward

Every action that our agent takes will result in a reward of some magnitude (positive or negative). The goal of our agent will be to maximize its reward in an enviornment. Sometimes the reward will be clear, for example if an agent performs an action which increases their score in the enviornment we could say they've recieved a positive reward. If the agent were to perform an action which results in them losing score or possibly dying in the enviornment then they would recieve a negative reward. 

---

## Q-Learning

Q-Learning is a simple yet quite powerful technique in machine learning that involves learning a matrix of action-reward values. This matrix is often reffered to as a Q-Table or Q-Matrix. The matrix is in shape (number of possible states, number of possible actions) where each value at matrix[n, m] represents the agents expected reward given they are in state n and take action m. The Q-learning algorithm defines the way we update the values in the matrix and decide what action to take at each state. The idea is that after a succesful training/learning of this Q-Table/matrix we can determine the action an agent should take in any state by looking at that states row in the matrix and taking the maximium value column as the action.

Let's say A1-A4 are the possible actions and we have 3 states represented by each row (state 1 - state 3).

![](https://i.imgur.com/P3bO5Hl.png)

If that was our Q-Table/matrix then the following would be the preffered actions in each state.
* State 1: A3
* State 2: A2
* State 3: A1

We can see that this is because the values in each of those columns are the highest for those states.

### Learning the Q-Table

Q-Table starts of with all 0 values. This is because the agent has yet to learn anything about the enviornment. 

Our agent learns by exploring the enviornment and observing the outcome/reward from each action it takes in each state. But how does it know what action to take in each state? There are two ways that our agent can decide on which action to take.
* Randomly picking a valid action.
* Using the current Q-Table to find the best action.

Near the beginning of our agents learning it will mostly take random actions in order to explore the enviornment and enter many different states. As it starts to explore more of the enviornment it will start to gradually rely more on it's learned values (Q-Table) to take actions. This means that as our agent explores more of the enviornment it will develop a better understanding and start to take "correct" or better actions more often. It's important that the agent has a good balance of taking random actions and using learned values to ensure it does get trapped in a local maximum. 

After each new action our agent wil record the new state (if any) that it has entered and the reward that it recieved from taking that action. These values will be used to update the Q-Table. The agent will stop taking new actions only once a certain time limit is reached or it has acheived the goal or reached the end of the enviornment. 

### Updating Q-Values

The formula for updating the Q-Table after each action is as follows:
![](https://i.imgur.com/zSwCDlJ.png)

* **α** stands for the **Learning Rate**.
* **γ** stands for the **Discount Factor**.

To perform updates on this table we will let the agent explpore the enviornment for a certain period of time and use each of its actions to make an update. Slowly we should start to notice the agent learning and choosing better actions. 

### Learning Rate α

The learning rate α is a numeric constant that defines how much change is permitted on each QTable update. A high learning rate means that each update will introduce a large change to the current state-action value. A small learning rate means that each update has a more subtle change. Modifying the learning rate will change how the agent explores the enviornment and how quickly it determines the final values in the QTable.

### Discount Factor γ

Discount factor also know as gamma (γ) is used to balance how much focus is put on the current and future reward. A high discount factor means that future rewards will be considered more heavily.

---

## Sources

* Violante, Andre. “Simple Reinforcement Learning: Q-Learning.” Medium, Towards Data Science, 1 July 2019, https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56.
* Openai. “Openai/Gym.” GitHub, https://github.com/openai/gym/wiki/FrozenLake-v0.

---
