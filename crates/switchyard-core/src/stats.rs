//! Statistics shared by scheduling forecasts and reports.

/// Nearest-rank percentile; empty samples remain absent rather than reporting zero.
pub fn percentile(mut values: Vec<f64>, quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(
        values[((values.len() as f64 * quantile).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1)],
    )
}
