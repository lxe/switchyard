//! Replay the checked-in analytical recipe with all reference policies.
use std::error::Error;
use switchyard::{POLICIES, Profile, Scenario, report, simulate};

fn main() -> Result<(), Box<dyn Error>> {
    let profiles: Vec<Profile> =
        serde_json::from_str(include_str!("../configs/quickstart/profiles.json"))?;
    let scenario: Scenario =
        serde_json::from_str(include_str!("../configs/quickstart/burst.json"))?;
    let runs = POLICIES
        .into_iter()
        .map(|policy| simulate(&profiles, &scenario, policy, 7))
        .collect::<Result<Vec<_>, _>>()?;
    print!("{}", report::markdown(&runs));
    Ok(())
}
