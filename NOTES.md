

Value:
    -Value weight is used in scoring/optimization: it contributes to total_value_destroyed and objective scoring
    -Freighter value_weight = 1
    -Escort value_weight = 
    -Tanker value_weight = 
    -Decoy value_weight = 
    -

Awesome, great.

Now: render_attack_animation.py

Fixes:
-Vastly expand the blue area so more of the sim can be seen. Right now it is a skinny rectangle. Essentially zoom out more
-Move the legend down so it does not overlap with the x axis label
-Add: for spine in plt.gca().spines.values(): spine.set_visible(False)
-If possible add the dynamic ship markers (the ships in this viz are not moving up but rather upper left ish)
-Can we increase the size of the convoy, Right now it is only three ships. Maybe make it similar to the historical layout?
-Remove black grid lines around the convoy
-Make torpedo lines red instead of grey


Update Static Example to:
- Make ships red instead of circles when hit
-Center the plot right, now it is pushed right due to axis labels
















for spine in plt.gca().spines.values():
    spine.set_visible(False)