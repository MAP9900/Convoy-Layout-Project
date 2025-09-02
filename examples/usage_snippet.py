import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) #Enables Code Runner to work

from convoy.physics import ShipState, Torpedo, hit_time

# Construct one ship and one torpedo and print the first hit time (if any).
ship = ShipState(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=20.0)
torp = Torpedo(
    x0=-800.0, y0=5.0,
    ux=1.0, uy=0.0,
    v_t=50.0,
    t_launch=0.0,
    max_run=3000.0,
)

print("First hit time:", hit_time(ship, torp))