use candle_core::{Device, Tensor};
use switchyard_engine::sampling::Greedy;

fn tied_maxima(device: &Device) {
    let mut values = vec![-1f32; 4096];
    values[11] = 3.;
    values[2048] = 3.;
    values[4096 - 1] = 3.;
    let logits = Tensor::from_vec(values, (1, 4096), device).unwrap();
    assert_eq!(
        Greedy::new(4096, device)
            .unwrap()
            .select(&logits)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap(),
        vec![11]
    );
}
#[test]
fn lowest_vocabulary_id_wins_ties() {
    tied_maxima(&Device::Cpu);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_ties_match_cpu_rule() {
    tied_maxima(&Device::new_cuda(0).unwrap());
}
