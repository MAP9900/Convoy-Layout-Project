# Convoy Layout & Attack Optimization — Design Framework

## 1. Operational Realism  
*Plausibility without overfitting history*

### Attack feasibility & detection constraints
- Escort exclusion zones that limit attacker approach regions.
- Feasible attack cones (abeam vs bow-on vs stern).
- Minimum and maximum viable firing ranges.
- Environment-dependent approach constraints (night vs day, visibility).

### Heterogeneous ships
- Multiple ship classes (freighter, tanker, escort, decoy).
- Different physical sizes (beam, hit radius).
- Different values or loss weights.
- Objectives based on value destroyed rather than raw hit count.

### Temporal realism
- Convoy heading changes over time (legs).
- Convoy-level zig-zag patterns (slow, coordinated).
- Time-dependent attack windows (attacks optimized for *when*, not just *where*).

---

## 2. Adversarial Decision-Making  
*Where this becomes an AI project*

### Defender as a policy
- Layout choice conditioned on threat priors.
- Different layouts for different expected attack styles.
- Potentially mixed or randomized layouts.

### Attacker tactics beyond a single salvo
- Decision to delay or abort attacks.
- Partial salvos.
- Edge-biased or asymmetric spreads.
- Multiple attack passes if feasible.

### Game-theoretic framing
- Defender layout ↔ attacker spread as strategic moves.
- Best-response analysis.
- Exploitability and approximate Nash equilibrium.
- Mixed strategies (randomized layouts or attack angles).

---

## 3. Learning, Robustness, and Uncertainty

### Robust optimization
- Optimize for CVaR / worst-case outcomes, not just mean hits.
- Layouts that remain effective under uncertainty.
- Attacks that dominate across parameter ranges.

### Surrogate modeling
ML models approximating simulation outcomes.

**Used for:**
- Fast optimization.
- Sensitivity analysis.
- Interpretability (feature importance).

### Reinforcement learning (advanced)
- Discrete action menus for stability.
- Self-play attacker/defender learning.
- Policies instead of fixed parameters.

---

## 4. Visualization & Interpretability  
*Critical for insight*

### Convoy layout visualizations
- Plan-view plots of ship positions.
- Color-coding by ship class or value.
- Footprint outlines for layout comparison.
- Overlays comparing historical vs optimized layouts.

### Attack visualizations
- Torpedo rays overlaid on convoy geometry.
- Hit locations and miss distances.
- Temporal animation of salvo firing and convoy motion.
- Density maps showing torpedo coverage across the convoy face.

### Comparative visual diagnostics
- Side-by-side layout comparisons.
- Before/after optimization views.
- Highlighting vulnerable alignment structures (rows/columns).

**These visuals are essential for:**
- Debugging geometry.
- Explaining results.
- Communicating insight beyond raw metrics.

---

## 5. Custom Layout Construction  
*Human-in-the-loop design*

### Parametric layout builders
- Rectangular, staggered, hex, jittered families.
- Footprint-constrained layouts.
- Aspect-ratio controlled designs.

### Manual / custom layouts
- User-specified ship coordinates.
- Scriptable layout templates.
- Hybrid layouts (e.g., dense core + sparse perimeter).

### Layout validation
- Minimum separation enforcement.
- Station-keeping feasibility checks.
- Automatic rejection of invalid layouts.

**This allows:**
- Human intuition to guide experiments.
- Testing “what-if” designs not discoverable via optimization alone.

---

## 6. Scientific Framing  
*What makes this research-grade*

### Explicit hypotheses
- “Staggered layouts reduce multi-hit probability under abeam fan spreads.”
- “Hexagonal layouts reduce average damage but increase tail risk.”
- “Escort exclusion zones shift optimal attack timing more than geometry.”

### Ablations
- Remove stagger → measure impact.
- Remove timing noise → measure impact.
- Fix attacker priors → measure impact.

### Counterfactual analysis
- Why historically optimal layouts may not have been adopted.
- Tradeoffs between safety, complexity, and command/control.

---

## 7. Stretch & Flagship Extensions
- Multi-U-boat coordinated attacks (wolfpacks).
- Convoy heterogeneity with high-value targets.
- Dynamic layouts that evolve during an engagement.
- Generative models for layout creation.
- Full experimental write-up with figures and ablation tables.