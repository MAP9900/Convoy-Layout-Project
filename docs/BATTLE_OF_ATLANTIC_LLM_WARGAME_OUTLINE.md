# Battle Of The Atlantic LLM Wargame (Brief Outline)

## Concept

Turn-based operational wargame where two LLM agents compete:
- `Allied Command`: deliver supplies from North America to Europe.
- `U-boat Command`: interdict convoys and reduce delivered tonnage.

The simulator resolves outcomes; LLMs choose bounded strategic actions.

## Why This Scope Works

- Keeps physics/combat abstract enough for a class timeline.
- Supports many seeded matches for statistical comparison.
- Avoids brittle free-form LLM behavior by requiring structured actions.

## MVP Game Design

### State (weekly turn)
- Atlantic map split into sectors/lane graph
- convoy schedules and cargo tonnage
- escort availability and readiness
- U-boat availability, fuel/endurance, patrol positions
- weather and intel uncertainty

### Agent Actions (JSON only)
- Allies:
  - route selection per convoy
  - escort allocation by convoy/sector
  - ASW focus priorities
- U-boats:
  - patrol zone assignment
  - wolfpack concentration target
  - attack timing/doctrine posture

### Turn Resolver
- movement and lane progression
- detection checks (stochastic, weather/intel conditioned)
- engagement outcome model (losses, damage, disruption)
- logistics update (arrivals, delays, replacements)

## Objectives

- Allies: maximize delivered tonnage, minimize losses and delay.
- U-boats: maximize tonnage sunk and route disruption.

## Evaluation Plan

1. Run many seeded games per matchup.
2. Compare:
   - LLM vs scripted baseline
   - prompt/doctrine variants
   - ablations (with/without uncertainty, with/without wolfpack bonus)
3. Report:
   - delivered tonnage
   - tonnage sunk
   - convoy survival rate
   - average delay
   - win rate by side

## Implementation Plan (2–3 Week Shape)

1. Build sector-map turn engine + deterministic artifact logging.
2. Define strict action schema and validation.
3. Add two baseline scripted commanders.
4. Integrate LLM commanders constrained to schema outputs.
5. Run tournament harness and produce summary plots/tables.

## Risks And Controls

- Risk: LLM action drift.
  - Control: schema validation + fallback default action.
- Risk: unstable results from small sample counts.
  - Control: fixed seed suites + confidence intervals.
- Risk: over-scoping combat realism.
  - Control: keep engagement model abstract and calibrated.

## Stretch Options

- hierarchical command (strategy planner + operations executor)
- fog-of-war memory and deceptive intel events
- limited diplomacy/coordination events (air cover, rerouting requests)
