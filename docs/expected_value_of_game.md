# Expected value of a game

`expected_value_of_game(game)` computes the expected value of a sequential
game for each row of a Polars expression.

A game is a list of states. A state is a list of action structs with these
fields:

- `probability`: probability of choosing the action;
- `value`: the action's cost or reward;
- `next_state`: index of the state reached by the action.

The first state is initial and the last state is final. The final state must
have no actions. The calculation proceeds backward from the final state. The
caller is responsible for ensuring the graph has no loops; requiring each
`next_state` to be greater than the current state index is sufficient and is
validated by the function.

```python
import polars as pl
from polars_list_math import expected_value_of_game

games = pl.DataFrame(
    {
        "game": [
            [
                [
                    {"probability": 0.5, "value": 2.0, "next_state": 1},
                    {"probability": 0.5, "value": 10.0, "next_state": 2},
                ],
                [{"probability": 1.0, "value": 3.0, "next_state": 2}],
                [],
            ]
        ]
    }
)

result = games.select(expected_value_of_game("game"))
```
