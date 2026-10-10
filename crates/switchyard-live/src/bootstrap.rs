//! Package an agent bootstrap as a provider-independent container request.
use crate::config::BootstrapConfig;
use switchyard_provider::{ContainerSpec, Port};
const SCRIPT: &str = r#"set -eu
mkdir -p /opt/switchyard
curl -fsSL --retry 3 "$SWITCHYARD_AGENT_URL" -o /opt/switchyard/agent
printf '%s  %s\n' "$SWITCHYARD_AGENT_SHA256" /opt/switchyard/agent | sha256sum -c --quiet
chmod 700 /opt/switchyard/agent
exec /opt/switchyard/agent
"#;

impl BootstrapConfig {
    pub(crate) fn container(
        &self,
        disk_gb: u64,
        ticket_url: String,
        agent_token: String,
    ) -> ContainerSpec {
        ContainerSpec {
            image: self.image.clone(),
            disk_gb,
            entrypoint: vec!["/bin/bash".into(), "-c".into()],
            command: vec![SCRIPT.into()],
            ports: vec![Port {
                number: 8080,
                http: true,
            }],
            environment: [
                ("SWITCHYARD_AGENT_URL".into(), self.agent_url.clone()),
                ("SWITCHYARD_AGENT_SHA256".into(), self.agent_sha256.clone()),
                ("SWITCHYARD_TICKET_URL".into(), ticket_url),
                ("SWITCHYARD_AGENT_TOKEN".into(), agent_token),
                ("HF_HUB_OFFLINE".into(), "1".into()),
                ("VLLM_NO_USAGE_STATS".into(), "1".into()),
            ]
            .into_iter()
            .collect(),
        }
    }
}
