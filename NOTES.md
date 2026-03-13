

Value:
    -Value weight is used in scoring/optimization: it contributes to total_value_destroyed and objective scoring
    -Freighter value_weight = 1
    -Escort value_weight = 
    -Tanker value_weight = 
    -Decoy value_weight = 
    -
Keep existing small visual defaults untouched (scenario_a + current _make_convoy).
Add a new RL convoy profile as a separate layout config (new scenario or profile registry entry), not by editing the baseline files.
Refactor visuals to accept a profile/scenario selector (CLI arg like --profile rl_large or --scenario scenario_rl) while defaulting to current small setup.

for spine in plt.gca().spines.values():
    spine.set_visible(False)

