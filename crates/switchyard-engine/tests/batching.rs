use candle_core::{Device, Tensor};
use switchyard_engine::{
    Result,
    config::{Config, DeviceConfig, Precision},
    engine::{Engine, Generate},
    models::{ModelFactory, ModelSession},
    telemetry::Telemetry,
};

struct Counter;
impl ModelFactory for Counter {
    fn session(&self) -> Box<dyn ModelSession> {
        Box::new(Counter)
    }
    fn device(&self) -> &Device {
        &Device::Cpu
    }
    fn vocab_size(&self) -> usize {
        16
    }
    fn eos_token(&self) -> u32 {
        15
    }
}
impl ModelSession for Counter {
    fn forward(&mut self, tokens: &Tensor, _position: usize) -> Result<Tensor> {
        let rows = tokens.to_vec2::<u32>()?;
        let mut logits = vec![0f32; rows.len() * 16];
        for (row, input) in rows.iter().enumerate() {
            logits[row * 16 + ((input.last().unwrap() + 1) % 16) as usize] = 1.;
        }
        Ok(Tensor::from_vec(logits, (rows.len(), 16), &Device::Cpu)?)
    }
}
fn engine() -> Engine {
    Engine::new(
        Config {
            model: "counter".into(),
            model_root: "unused".into(),
            device: DeviceConfig::Cpu,
            dtype: Precision::F32,
            max_context: 64,
            max_weight_bytes: 1024,
            max_batch_size: 4,
            queue_capacity: 8,
            batch_wait_ms: 0,
        },
        Box::new(Counter),
        Telemetry::default(),
    )
}
#[test]
fn generated_workloads_preserve_order_output_bounds_and_eos() {
    let engine = engine();
    let mut random = 71u64;
    for _ in 0..100 {
        let mut requests = Vec::new();
        for row in 0..9 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            requests.push(Generate {
                prompt: vec![(random % 16) as u32; row % 5 + 1],
                max_tokens: row % 7 + 1,
                ignore_eos: row % 2 == 0,
            });
        }
        let output = engine.generate(&requests).unwrap();
        for (r, out) in requests.iter().zip(output) {
            let mut expected = Vec::new();
            let mut token = *r.prompt.last().unwrap();
            for _ in 0..r.max_tokens {
                token = (token + 1) % 16;
                expected.push(token);
                if token == 15 && !r.ignore_eos {
                    break;
                }
            }
            assert_eq!(out.token_ids, expected);
            assert_eq!(out.prompt_tokens, r.prompt.len());
            assert!(out.token_latency_s.windows(2).all(|t| t[0] <= t[1]));
        }
    }
}
#[test]
fn cancellation_stops_one_row_without_truncating_the_other() {
    let requests = vec![
        Generate {
            prompt: vec![1],
            max_tokens: 4,
            ignore_eos: true
        };
        2
    ];
    let output = engine()
        .generate_observed(&requests, |row, _, _, _| row != 0)
        .unwrap();
    assert_eq!(output[0].token_ids, vec![2]);
    assert_eq!(output[0].finish_reason, "cancelled");
    assert_eq!(output[1].token_ids, vec![2, 3, 4, 5]);
}
#[test]
fn admission_rejects_empty_invalid_and_overflowing_requests() {
    for request in [
        Generate {
            prompt: vec![],
            max_tokens: 1,
            ignore_eos: false,
        },
        Generate {
            prompt: vec![16],
            max_tokens: 1,
            ignore_eos: false,
        },
        Generate {
            prompt: vec![1],
            max_tokens: usize::MAX,
            ignore_eos: false,
        },
        Generate {
            prompt: vec![1],
            max_tokens: 0,
            ignore_eos: false,
        },
    ] {
        assert!(engine().generate(&[request]).is_err());
    }
}
