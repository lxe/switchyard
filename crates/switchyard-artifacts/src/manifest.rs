use crate::{Error, Result, paths};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub(crate) const FORMAT: &str = "switchyard-artifact-v1";
pub(crate) const RECEIPT: &str = ".switchyard-artifact-id";

/// Caller-defined artifact classification, compatibility labels, and provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub kind: String,
    #[serde(default)]
    pub compatibility: BTreeMap<String, String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

/// Exact filesystem entries. Ownership and timestamps are normalized in archives.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Entry {
    File {
        path: String,
        mode: u32,
        size: u64,
        sha256: String,
    },
    Directory {
        path: String,
        mode: u32,
    },
    Symlink {
        path: String,
        mode: u32,
        target: String,
    },
}

impl Entry {
    pub fn path(&self) -> &str {
        match self {
            Self::File { path, .. } | Self::Directory { path, .. } | Self::Symlink { path, .. } => {
                path
            }
        }
    }
    pub fn mode(&self) -> u32 {
        match self {
            Self::File { mode, .. } | Self::Directory { mode, .. } | Self::Symlink { mode, .. } => {
                *mode
            }
        }
    }
    pub(crate) fn size(&self) -> u64 {
        match self {
            Self::File { size, .. } => *size,
            _ => 0,
        }
    }
}

/// One independently compressed tar stream. `unpacked_size` includes tar headers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chunk {
    pub name: String,
    pub sha256: String,
    pub size: u64,
    pub unpacked_size: u64,
    /// Indices into the manifest entries, in this tar stream's extraction order.
    pub entries: Vec<usize>,
}

/// Identity covers format, metadata, ordered filesystem entries, and compressed chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub id: String,
    pub metadata: Metadata,
    pub entries: Vec<Entry>,
    pub chunks: Vec<Chunk>,
}

/// Storage locations are supplied by integration code; URLs are never included in identity.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactTicket {
    pub manifest: Manifest,
    /// Map each `Chunk::name` to its downloadable HTTP(S) URL.
    pub urls: BTreeMap<String, String>,
}

impl Manifest {
    pub(crate) fn identity(&self) -> Result<String> {
        let bytes =
            serde_json::to_vec(&(&self.format, &self.metadata, &self.entries, &self.chunks))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != FORMAT || !valid_digest(&self.id) || self.id != self.identity()? {
            return Err(Error::Invalid("format or manifest identity".into()));
        }
        self.validate_structure()
    }

    // Trusted startup still enforces filesystem safety, coverage and declared
    // bounds, but relies on its caller's pinned manifest instead of rehashing it.
    pub(crate) fn validate_structure(&self) -> Result<()> {
        if self.format != FORMAT || !valid_digest(&self.id) {
            return Err(Error::Invalid("format or manifest identity".into()));
        }
        self.validate_entries()?;
        self.validate_chunks()
    }

    pub(crate) fn validate_entries(&self) -> Result<()> {
        if self.metadata.kind.trim().is_empty()
            || self.metadata.kind.contains('\0')
            || self
                .metadata
                .compatibility
                .keys()
                .chain(self.metadata.attributes.keys())
                .any(|key| key.trim().is_empty() || key.contains('\0'))
            || self.entries.len() > 1_000_000
            || self.chunks.len() > 1_000_000
        {
            return Err(Error::Invalid("metadata or entry limits".into()));
        }
        let mut entries = BTreeMap::new();
        let mut previous = None;
        for entry in &self.entries {
            paths::relative(entry.path())?;
            if entry.path() == RECEIPT || entry.path().starts_with(&format!("{RECEIPT}/")) {
                return Err(Error::Invalid("reserved artifact receipt path".into()));
            }
            if previous.is_some_and(|path: &str| path >= entry.path()) {
                return Err(Error::Invalid("entries must be sorted and unique".into()));
            }
            previous = Some(entry.path());
            if entry.mode() & !0o777 != 0 || entry.mode() & 0o400 == 0 {
                return Err(Error::Invalid(
                    "unsafe or unreadable entry permissions".into(),
                ));
            }
            match entry {
                Entry::File { sha256, .. } if !valid_digest(sha256) => {
                    return Err(Error::Invalid("file digest".into()));
                }
                Entry::Directory { mode, .. } if mode & 0o100 == 0 => {
                    return Err(Error::Invalid("unsearchable directory".into()));
                }
                Entry::Symlink { mode, target, path } => {
                    if *mode != 0o777 {
                        return Err(Error::Invalid("symlink mode must be 0777".into()));
                    }
                    paths::link_target(path, target)?;
                }
                _ => {}
            }
            entries.insert(entry.path(), entry);
        }
        for entry in &self.entries {
            let path = std::path::Path::new(entry.path());
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty())
                && !matches!(
                    entries.get(parent.to_str().unwrap()),
                    Some(Entry::Directory { .. })
                )
            {
                return Err(Error::Invalid(
                    "entry ancestor is not an explicit directory".into(),
                ));
            }
            if let Entry::Symlink { path, target, .. } = entry {
                let target = paths::link_target(path, target)?;
                if !target.is_empty() && !entries.contains_key(target.as_str()) {
                    return Err(Error::Invalid(
                        "symlink target is absent from artifact".into(),
                    ));
                }
                let _ = resolve_link(entry.path(), entry, &entries)?;
            }
        }
        Ok(())
    }

    fn validate_chunks(&self) -> Result<()> {
        let mut names = BTreeSet::new();
        let mut covered = BTreeSet::new();
        let mut compressed = 0_u64;
        let mut unpacked = 0_u64;
        for chunk in &self.chunks {
            if !valid_digest(&chunk.sha256)
                || chunk.name != format!("{}.tar.zst", chunk.sha256)
                || !names.insert(&chunk.name)
                || chunk.size == 0
                || chunk.unpacked_size < 1024
                || chunk.entries.is_empty()
            {
                return Err(Error::Invalid("chunk identity, size, or coverage".into()));
            }
            compressed = compressed
                .checked_add(chunk.size)
                .ok_or_else(|| Error::Invalid("compressed size overflow".into()))?;
            unpacked = unpacked
                .checked_add(chunk.unpacked_size)
                .ok_or_else(|| Error::Invalid("unpacked size overflow".into()))?;
            for index in &chunk.entries {
                if *index >= self.entries.len() || !covered.insert(*index) {
                    return Err(Error::Invalid("chunk entry coverage".into()));
                }
            }
        }
        if covered.len() != self.entries.len() {
            return Err(Error::Invalid("missing chunk entries".into()));
        }
        Ok(())
    }

    /// Require exact values for every supplied compatibility key; extra labels are allowed.
    pub fn check_compatibility(&self, required: &BTreeMap<String, String>) -> Result<()> {
        self.validate()?;
        for (key, value) in required {
            if self.metadata.compatibility.get(key) != Some(value) {
                return Err(Error::Compatibility(key.clone()));
            }
        }
        Ok(())
    }

    /// Resolve an existing declared path to its final file or directory entry.
    ///
    /// This validates the manifest and follows internal symlinks using filesystem
    /// semantics. The input must be a canonical relative path naming an explicit
    /// manifest entry; absent or unsafe input returns [`Error::Invalid`]. A link
    /// resolving to the artifact root returns `None`, because the root is not a
    /// declared entry. The returned permissions belong to the final entry rather
    /// than a symlink, whose mode does not establish target permissions.
    pub fn resolve_entry(&self, path: &str) -> Result<Option<&Entry>> {
        self.validate()?;
        paths::relative(path)?;
        let entries = self
            .entries
            .iter()
            .map(|entry| (entry.path(), entry))
            .collect::<BTreeMap<_, _>>();
        let entry = entries
            .get(path)
            .copied()
            .ok_or_else(|| Error::Invalid("requested entry is absent from artifact".into()))?;
        resolve_link(path, entry, &entries)
    }
}

/// Resolve declared links with filesystem semantics, including links encountered before `..`.
/// Lexical containment alone is insufficient: `alias/../file` escapes if `alias` points at root.
fn resolve_link<'a>(
    path: &str,
    entry: &'a Entry,
    entries: &BTreeMap<&str, &'a Entry>,
) -> Result<Option<&'a Entry>> {
    let Entry::Symlink { target, .. } = entry else {
        return Ok(Some(entry));
    };
    let mut resolved = path.split('/').collect::<Vec<_>>();
    resolved.pop();
    let mut remaining = target.split('/').collect::<VecDeque<_>>();
    let mut links = 0;
    while let Some(component) = remaining.pop_front() {
        match component {
            "" | "." => continue,
            ".." => {
                if resolved.pop().is_none() {
                    return Err(Error::Invalid(
                        "resolved symlink escapes artifact root".into(),
                    ));
                }
                continue;
            }
            _ => resolved.push(component),
        }
        let name = resolved.join("/");
        match entries.get(name.as_str()) {
            Some(Entry::Symlink { target, .. }) => {
                links += 1;
                if links > 40 {
                    return Err(Error::Invalid(
                        "symlink cycle or excessive link depth".into(),
                    ));
                }
                resolved.pop();
                for component in target.split('/').rev() {
                    remaining.push_front(component);
                }
            }
            Some(Entry::Directory { .. }) => {}
            Some(Entry::File { .. }) if remaining.is_empty() => {}
            Some(Entry::File { .. }) => {
                return Err(Error::Invalid("symlink traverses a file".into()));
            }
            None => return Err(Error::Invalid("resolved symlink target is absent".into())),
        }
    }
    if resolved.is_empty() {
        Ok(None)
    } else {
        entries
            .get(resolved.join("/").as_str())
            .copied()
            .map(Some)
            .ok_or_else(|| Error::Invalid("resolved symlink target is absent".into()))
    }
}

pub(crate) fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
