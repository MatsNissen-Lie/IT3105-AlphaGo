# IT3105-AlphaGo

Vi lager MCTS og AlphaGo for å spille HEX.

MCTS har rollouts og en critic

Vi bruker rollouts til å begynne med, men vi belager oss mer og mer på the critic ettersom den blir bedre.

MCTS:
Balancing exploration and exploitation

- UBC1: balanserer exploration og exploitation ved å ta hensyn til hvor mange ganger en node har blitt besøkt og hvor stor verdien er. Blir brukt til å velge hvilken node vi skal utforske videre.

- Rollouts: Utfører en simulering av et spill fra en node til en terminal node. Velger neste node basert på Greedy epsilon policy. Det vil si at vi velger en tilfeldig node med sannsynlighet epsilon, og ellers velger vi den noden som gir høyest verdi basert på estimater fra neural networket.
