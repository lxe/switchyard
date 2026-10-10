# Analytical inference recipe

This recipe describes twelve requests for `demo-model`, one initially warm small instance, and two synthetic GPU offerings. Capacity, startup, and prices are analytical assumptions used to demonstrate replay and routing. They do not describe a real model or rentable GPU.

Run `cargo run --locked --release --example replay` from the repository root to compare all policies. `profiles.json` is the offering catalog, `burst.json` is the offline trace, and `environment.json` is the HTTP provider configuration. The root README shows both workflows.

Every request supplies 512 input tokens, an estimated 64 output tokens, and a hard bound of 128. The physical fixture produces 96 output tokens; this field is absent from scheduler observations. The job requests 200 delivered output tokens per second and a twenty-second completion budget. A missed goal is reported as a policy outcome.
