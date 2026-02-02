

Value:
    -Value weight is used in scoring/optimization: it contributes to total_value_destroyed and objective scoring
    -Freighter value_weight = 1
    -Escort value_weight = 
    -Tanker value_weight = 
    -Decoy value_weight = 
    -

plot_attack_once.py now!
-Can we make the graphic bigger and focus more on the convoy? The torpedos go out to 8000 m (which may be historically accurate) but is not needed for the plot, just focus on the convoy
-If possible add the new ship markers instead of the circles for the convoys, but insure hit markers (red circles) still work
-Move the legend down so it does not overlap with the x axis
- add: for spine in plt.gca().spines.values():
    spine.set_visible(False)
-Change Title to "Static Attack Example"
-Change facecolor='lightgrey'


-Expand the visual to also include the point of origin of the torpedos and add a maker for the submarine (add this to legend too, just make it a circle)
-Remove the grid around the convoy
-Either the physics are wrong or the ship markers are too small/not to scale as the the right middle ships are claimed to be hit but the torpedo lines does not intersect them


for spine in plt.gca().spines.values():
    spine.set_visible(False)