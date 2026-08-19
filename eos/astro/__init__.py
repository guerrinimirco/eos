"""Stellar structure and oscillations: what an equation of state is FOR.

This layer consumes tables and arrays produced by the models and the composite
engines and turns them into observables. It never imports model internals, and
no model imports it (CLAUDE.md section 1):

    general/  ->  models  ->  composite engines  ->  astro/

    tov/      stellar structure -- TOV, tidal deformability, crust attachment,
              and uniformly rotating models through the RNS backend
    gmode/    composition g-modes of the resulting stars
"""
