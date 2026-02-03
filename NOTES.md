

Value:
    -Value weight is used in scoring/optimization: it contributes to total_value_destroyed and objective scoring
    -Freighter value_weight = 1
    -Escort value_weight = 
    -Tanker value_weight = 
    -Decoy value_weight = 
    -

New flag and spec in render_attack_animation.py: apply_hit_slowdown = False  decay_rate is used to determine how much ships slow down by

Independent motion: ship_positions_at now integrates each ship’s position forward over time using the convoy heading (zig‑zag included), rather than rotating a rigid formation. Rigid mode still exists as an explicit option (motion="rigid"), but the default is independent.
Sim alignment: all dynamic hit logic now uses the independent motion path so the visuals and sim stay in lock‑step.
Slowdown flag: render_attack_animation.py now has a toggle and an exponential decay profile in place.



Back To: render_attack_animation.py
-Either the hit recognization is wrong or ships are way out of scale as torpedos appear to pass through the middle of ships but no red circle indicating hits appear. Again I want the circle to appear in all remaining frames to know which ships were hit.
-Also perhaps the red circle can also change colors indicating number of hits. Start with a light red and get darker. Have four shades of red for this. 
-Also to confirm a torpedo can only hit a ship once correct? If so ideally once it hits a torpedo, the red line stops as to indicate a hit

Ok two issues:
-For starters it breaks between frames 299 and 300. The rotation of the ships switches and the hit ship completly changes. It goes from 4 ships tall, 3 wide to 3 tall 4 wide at frame 300.
-Second, despite the zig, the ships do not appear to be moving "forward" at all times during the zig-zag. At times the ships mve back ward like the zig is just rotating them on their center axis. All ships need to moving forward at all times. Do not fix this portion yet as I want your feedback on this. Also give thoughts on giving ships individual speeds so hit ships can slow down. Again the core goal is realism. 




for spine in plt.gca().spines.values():
    spine.set_visible(False)