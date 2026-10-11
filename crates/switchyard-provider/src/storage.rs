//! S3-compatible storage uses the AWS SigV4 signer; nodes receive expiring GET URLs.
mod upload;
use crate::{Error, Result, StorageConfig};
use aws_credential_types::Credentials;
use aws_sigv4::{
    http_request::{
        PercentEncodingMode, SignableBody, SignableRequest, SignatureLocation, SigningSettings,
        UriPathNormalizationMode, sign,
    },
    sign::v4,
};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use reqwest::{Client, Method};

// SigV4 S3 signing consumes a URI whose segments are already RFC 3986 encoded.
const SEGMENT: AsciiSet = NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');
use std::time::{Duration, SystemTime};

#[derive(Clone)]
pub struct Storage {
    config: StorageConfig,
    credentials: Credentials,
    client: Client,
}
impl Storage {
    pub fn new(config: &StorageConfig) -> Result<Self> {
        let ak = std::env::var(&config.access_key_env)
            .map_err(|_| Error::Config(format!("missing {}", config.access_key_env)))?;
        let sk = std::env::var(&config.secret_key_env)
            .map_err(|_| Error::Config(format!("missing {}", config.secret_key_env)))?;
        Self::with_keys(config, ak, sk)
    }
    pub fn with_keys(config: &StorageConfig, ak: String, sk: String) -> Result<Self> {
        config.validate()?;
        if ak.is_empty() || sk.is_empty() {
            return Err(Error::Config("empty storage credential".into()));
        }
        Ok(Self {
            config: config.clone(),
            credentials: Credentials::new(ak, sk, None, None, "environment"),
            client: Client::builder()
                .user_agent("Switchyard/0.1")
                .timeout(Duration::from_secs(45))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|_| Error::Transport("storage client".into()))?,
        })
    }
    fn url(&self, key: &str) -> Result<reqwest::Url> {
        if key.is_empty()
            || key
                .split('/')
                .any(|part| matches!(part, "" | "." | "..") || part.chars().any(char::is_control))
        {
            return Err(Error::Config(
                "object keys must be relative paths with nonempty components".into(),
            ));
        }
        let mut url = reqwest::Url::parse(&self.config.endpoint)
            .map_err(|_| Error::Config("invalid storage endpoint".into()))?;
        let base_path = url.path().trim_end_matches('/');
        let encoded_key = key
            .split('/')
            .map(|part| utf8_percent_encode(part, &SEGMENT).to_string())
            .collect::<Vec<_>>()
            .join("/");
        let path = format!("{base_path}/{}/{encoded_key}", self.config.bucket);
        url.set_path(&path);
        Ok(url)
    }
    /// Sign an expiring GET capability. The returned URL contains credentials and must not be logged.
    pub fn presign(&self, key: &str) -> Result<String> {
        self.presign_at(key, SystemTime::now())
    }
    fn presign_at(&self, key: &str, time: SystemTime) -> Result<String> {
        let mut request = http::Request::builder()
            .method("GET")
            .uri(self.url(key)?.as_str())
            .body(())
            .map_err(|_| Error::Config("invalid object key".into()))?;
        let mut settings = SigningSettings::default();
        settings.signature_location = SignatureLocation::QueryParams;
        settings.expires_in = Some(Duration::from_secs(self.config.url_lifetime_s));
        settings.percent_encoding_mode = PercentEncodingMode::Single;
        settings.uri_path_normalization_mode = UriPathNormalizationMode::Disabled;
        let identity = self.credentials.clone().into();
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.config.region)
            .name("s3")
            .time(time)
            .settings(settings)
            .build()
            .map_err(|_| Error::Config("storage signing settings".into()))?
            .into();
        let signable = SignableRequest::new(
            "GET",
            request.uri().to_string(),
            std::iter::empty(),
            SignableBody::UnsignedPayload,
        )
        .map_err(|_| Error::Config("signable URL".into()))?;
        let (instructions, _) = sign(signable, &params)
            .map_err(|_| Error::Transport("storage signing".into()))?
            .into_parts();
        instructions.apply_to_request_http1x(&mut request);
        Ok(request.uri().to_string())
    }
    async fn request(&self, method: Method, key: &str, body: Vec<u8>) -> Result<reqwest::Response> {
        let url = self.url(key)?;
        let identity = self.credentials.clone().into();
        let mut settings = SigningSettings::default();
        settings.percent_encoding_mode = PercentEncodingMode::Single;
        settings.uri_path_normalization_mode = UriPathNormalizationMode::Disabled;
        settings.payload_checksum_kind = aws_sigv4::http_request::PayloadChecksumKind::XAmzSha256;
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.config.region)
            .name("s3")
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|_| Error::Config("storage signing settings".into()))?
            .into();
        let signable = SignableRequest::new(
            method.as_str(),
            url.as_str(),
            std::iter::empty(),
            SignableBody::Bytes(&body),
        )
        .map_err(|_| Error::Config("signable object".into()))?;
        let (instructions, _) = sign(signable, &params)
            .map_err(|_| Error::Transport("storage signing".into()))?
            .into_parts();
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in instructions.headers() {
            let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| Error::Config("signed header".into()))?;
            let mut value = reqwest::header::HeaderValue::from_str(value)
                .map_err(|_| Error::Config("signed header".into()))?;
            value.set_sensitive(true);
            headers.insert(name, value);
        }
        let response = self
            .client
            .request(method, url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(|_| Error::Transport("object request".into()))?;
        if response.status().is_success() {
            Ok(response)
        } else {
            Err(Error::Http {
                status: response.status().as_u16(),
                operation: "object request".into(),
            })
        }
    }
    pub async fn get_json<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<T> {
        self.request(Method::GET, key, vec![])
            .await?
            .json()
            .await
            .map_err(|_| Error::Response("object JSON".into()))
    }
    pub async fn put_json<T: serde::Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let bytes =
            serde_json::to_vec(value).map_err(|_| Error::Config("object serialization".into()))?;
        self.request(Method::PUT, key, bytes).await?;
        Ok(())
    }
    /// Deletion is idempotent, including a provider's explicit object-not-found response.
    pub async fn delete(&self, key: &str) -> Result<()> {
        match self.request(Method::DELETE, key, vec![]).await {
            Ok(_) | Err(Error::Http { status: 404, .. }) => Ok(()),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn store() -> Storage {
        Storage::with_keys(
            &StorageConfig {
                endpoint: "https://objects.example.test".into(),
                region: "auto".into(),
                bucket: "bucket".into(),
                access_key_env: "R2_ACCESS_KEY".into(),
                secret_key_env: "R2_SECRET_KEY".into(),
                url_lifetime_s: 900,
            },
            "example-access".into(),
            "example-secret".into(),
        )
        .unwrap()
    }
    #[test]
    fn presigned_url_matches_independently_calculated_sigv4_vector() {
        // Fixed S3 canonical request, checked with Python's hashlib/hmac independently of aws-sigv4.
        // GET /bucket/models/a%20b%2Bc%25.bin, host objects.example.test, UNSIGNED-PAYLOAD.
        let signed = store()
            .presign_at(
                "models/a b+c%.bin",
                SystemTime::UNIX_EPOCH + Duration::from_secs(1_778_416_496),
            )
            .unwrap();
        let url = reqwest::Url::parse(&signed).unwrap();
        let query: std::collections::BTreeMap<_, _> = url
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        assert_eq!(
            query["X-Amz-Signature"],
            "654ec91dba9757478962fad9e996d1ec6fbe402b30af6c19b418cde3af1f9464"
        );
        assert_eq!(
            query["X-Amz-Credential"],
            "example-access/20260510/auto/s3/aws4_request"
        );
        assert_eq!(query["X-Amz-Date"], "20260510T123456Z");
        assert_eq!(query["X-Amz-Expires"], "900");
        assert_eq!(query["X-Amz-SignedHeaders"], "host");
        assert!(!signed.contains("example-secret"));
    }
    #[test]
    fn object_keys_do_not_silently_normalize_to_different_objects() {
        for key in [
            "",
            "../manifest.json",
            "models/../manifest.json",
            "/manifest.json",
            "models//chunk",
            "models/./chunk",
            "models/chunk/",
        ] {
            assert!(store().presign(key).is_err(), "{key}");
        }
        assert!(store().presign("models/a b+c%.bin").is_ok());
    }
}
