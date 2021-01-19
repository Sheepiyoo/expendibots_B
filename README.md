# Expendibots: Part B
COMP30024 - Artificial Intelligence
Project Part B

This project was completed as a part of COMP30024 Artificial Intelligence subject at the University of Melbourne.
The aim of the project was to create an AI bot that would be able to win against another AI at a game called "Expendibots" without exceeding the time and space limitations of the project.
The report provides an overview of the methods that were implemented in our AI bot, including history heuristics, transpose tables, quiescence search and TDLeaf search, among other things. Overall, the final grade we received for this project was 90%.

Report link: https://docs.google.com/document/d/1FjtBbI_JmQxoeeEBA77XrQryH1CaY_etR5grYL4vsPk/edit?usp=sharing

AIMA link: https://github.com/aimacode/aima-python

Research link: https://docs.google.com/document/d/1bTRwBMd7buBIY3Xg-04X7ZsGW1kFUBIdj2D0Un35nL4/edit?usp=sharing

## Start with basic tests - random move
- [ ] Implement __init__(self, colour)
  * [ ] Set up internal representation
    
- [ ] Implement action(self)
  * [ ] Generate actions
  * [ ] Filter for valid actions
  * [ ] Select actions
   
- [ ] Implement update(self, colour, action)
  * [ ] Update our board
  
## Implement alpha-beta pruning
