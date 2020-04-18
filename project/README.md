## RLRL
Code for the rocket league reinforcement learning agent, submitted as part of CPSC533V.

In this folder, there are four notebooks:
1.  `behavioural_clone.ipynb` trains a network `f : S ⟶ A ` to predict  expert behaviour given their current state.
2.  `Deep_Q_Network.ipynb` trains a network `f : (S ✕ A) ⟶ ℝ` to predict Q values given a (state, action) pair.
3.  `artificial_rewards.ipynb` was used to define rewards from state and actions, and see how expert and in-game bots perform against those rewards.
4.  `future_predict.ipynb` was a notebook used to train `f : (S ✕ A) ⟶ S `  to try to predict the next state given the previous state and action. This was not used in the end.

**RLBot code in [`botSrc/CPSC533V/src`](https://github.com/padster/CPSC533V/tree/master/project/botSrc/CPSC533V/src)**:
This folder contains all the bot code needed for the [RLBot](https://www.rlbot.org/) framework to run the agent within the game itself. Of interest is [bot.py](https://github.com/padster/CPSC533V/blob/master/project/botSrc/CPSC533V/src/bot.py), which runs the trained agents by loading saved models, and translates the results into actions.

**code in [`lib/`](https://github.com/padster/CPSC533V/tree/master/project/lib)**:
Here is placed all common code, shared between the notebooks and/or the agent bot source. It includes logic to load, impute and normalize data from replay files, as well as calculations of artificial rewards, and the definitions of the two types of models used.

**Other**
`models/` includes the saved parameter files for the three models used by the agent.  
Videos of their behaviour is available on [Google Drive](https://drive.google.com/drive/folders/1DOPRF4nJia_EckekpUNma4cO5F-P_rqC).  
Finally, a .pdf is included describing the entire project.

