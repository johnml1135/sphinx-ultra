use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexDocument {
    pub schema_version: u32,
    pub generator: String,
    pub profiles: BTreeMap<String, ProfileRecord>,
    pub cases: Vec<CaseRecord>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileRecord {
    pub sphinx: String,
    pub docutils: String,
    pub needs_version: Option<String>,
    pub needs_commit: Option<String>,
    pub needs_tree: Option<String>,
    pub lock_path: String,
    pub lock_sha256: String,
    pub determinism_shims: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginRecord {
    pub source_set: String,
    pub origin_path: String,
    pub pytest_node_ids: Vec<String>,
    pub variants_not_captured: bool,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FileStorage {
    Input,
    Ref,
    Blob,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    pub logical_path: String,
    pub storage: FileStorage,
    pub storage_path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CaseStatus {
    Built,
    BuildError,
    ReferenceCrash,
    ExcludedNetwork,
    ExcludedPlantuml,
}

impl CaseStatus {
    pub fn is_excluded(self) -> bool {
        matches!(self, Self::ExcludedNetwork | Self::ExcludedPlantuml)
    }

    pub fn is_runnable(self) -> bool {
        matches!(self, Self::Built | Self::BuildError)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseRecord {
    pub profile: String,
    pub source_set: String,
    pub case_id: String,
    pub status: CaseStatus,
    pub exit_code: Option<i32>,
    pub warnings: String,
    pub excluded_reason: Option<String>,
    pub origin: OriginRecord,
    pub input_files: Vec<FileRecord>,
    pub input_sha256: String,
    pub tree_sha256: String,
    pub files: Vec<FileRecord>,
    pub needs_json: Option<FileRecord>,
    pub needs_status: Option<CaseStatus>,
    pub needs_exit_code: Option<i32>,
    pub needs_warnings: Option<String>,
}

impl IndexDocument {
    pub fn from_value(value: Value) -> Result<Self, String> {
        let document: Self = serde_json::from_value(value).map_err(|error| error.to_string())?;
        document.validate_shape()?;
        Ok(document)
    }

    pub fn load(index_path: &Path) -> Result<Self, String> {
        let bytes = fs::read(index_path)
            .map_err(|error| format!("read {}: {error}", index_path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse {}: {error}", index_path.display()))?;
        let document = Self::from_value(value)?;
        let profile_root = index_path
            .parent()
            .ok_or_else(|| format!("index has no parent: {}", index_path.display()))?;
        document.validate_files(profile_root)?;
        Ok(document)
    }

    pub fn validate_shape(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported schema_version {}",
                self.schema_version
            ));
        }
        if self.generator != "html-oracle/1" {
            return Err(format!("unexpected generator {:?}", self.generator));
        }
        for profile_name in self.profiles.keys() {
            if !matches!(profile_name.as_str(), "core" | "local_needs") {
                return Err(format!("unknown profile {profile_name:?}"));
            }
        }
        for (profile_name, profile) in &self.profiles {
            validate_sha256(
                &profile.lock_sha256,
                &format!("profile {profile_name} lock_sha256"),
            )?;
            validate_relative_path(&profile.lock_path, "lock_path")?;
        }

        let mut previous_key: Option<(&str, &str, &str)> = None;
        let mut keys = BTreeSet::new();
        for case in &self.cases {
            let key = (
                case.profile.as_str(),
                case.source_set.as_str(),
                case.case_id.as_str(),
            );
            if !keys.insert(key) {
                return Err(format!("duplicate case key {}/{}/{}", key.0, key.1, key.2));
            }
            if let Some(previous) = previous_key {
                if key < previous {
                    return Err(format!(
                        "cases are not sorted: {}/{}/{} follows {}/{}/{}",
                        key.0, key.1, key.2, previous.0, previous.1, previous.2
                    ));
                }
            }
            previous_key = Some(key);

            self.profiles
                .get(&case.profile)
                .ok_or_else(|| format!("case references unknown profile {:?}", case.profile))?;
            if case.origin.source_set != case.source_set {
                return Err(format!(
                    "origin source_set {:?} does not match case source_set {:?}",
                    case.origin.source_set, case.source_set
                ));
            }
            validate_case(case, profile_name_is_local_needs(&case.profile))?;
        }
        Ok(())
    }

    pub fn validate_files(&self, profile_root: &Path) -> Result<(), String> {
        let mut expected = BTreeSet::new();
        for case in &self.cases {
            let input_entries = case
                .input_files
                .iter()
                .map(|record| self.validate_file_record(profile_root, record, FileStorage::Input))
                .collect::<Result<Vec<_>, _>>()?;
            let output_entries = case
                .files
                .iter()
                .map(|record| self.validate_file_record(profile_root, record, record.storage))
                .collect::<Result<Vec<_>, _>>()?;
            if case.status.is_excluded() {
                continue;
            }
            let input_hash = canonical_hash(&input_entries);
            if input_hash != case.input_sha256 {
                return Err(format!(
                    "input_sha256 mismatch for {}/{}/{}: expected {}, got {}",
                    case.profile, case.source_set, case.case_id, case.input_sha256, input_hash
                ));
            }
            let tree_hash = canonical_hash(&output_entries);
            if tree_hash != case.tree_sha256 {
                return Err(format!(
                    "tree_sha256 mismatch for {}/{}/{}: expected {}, got {}",
                    case.profile, case.source_set, case.case_id, case.tree_sha256, tree_hash
                ));
            }
            for record in case.input_files.iter().chain(case.files.iter()) {
                expected.insert(record.storage_path.clone());
            }
            if let Some(record) = &case.needs_json {
                self.validate_file_record(profile_root, record, FileStorage::Ref)?;
                expected.insert(record.storage_path.clone());
            }
        }
        validate_reverse_artifacts(profile_root, &expected)
    }

    fn validate_file_record(
        &self,
        profile_root: &Path,
        record: &FileRecord,
        expected_storage: FileStorage,
    ) -> Result<(String, String), String> {
        if record.storage != expected_storage {
            return Err(format!(
                "storage kind mismatch for {}: expected {:?}, got {:?}",
                record.logical_path, expected_storage, record.storage
            ));
        }
        validate_relative_path(&record.logical_path, "logical_path")?;
        validate_relative_path(&record.storage_path, "storage_path")?;
        validate_sha256(&record.sha256, &format!("{} sha256", record.logical_path))?;
        let expected_prefix = match record.storage {
            FileStorage::Input => "inputs/",
            FileStorage::Ref => "refs/",
            FileStorage::Blob => "blobs/",
        };
        if !record.storage_path.starts_with(expected_prefix) {
            return Err(format!(
                "{} storage path must start with {expected_prefix:?}",
                record.logical_path
            ));
        }
        let path = profile_root.join(
            record
                .storage_path
                .replace('/', &std::path::MAIN_SEPARATOR.to_string()),
        );
        reject_symlink_components(profile_root, &path)?;
        let metadata = fs::metadata(&path)
            .map_err(|error| format!("read metadata {}: {error}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!("{} is not a regular file", path.display()));
        }
        if metadata.len() != record.size {
            return Err(format!(
                "size mismatch for {}: ledger {}, file {}",
                record.logical_path,
                record.size,
                metadata.len()
            ));
        }
        let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let actual_hash = sha256_hex(&bytes);
        if actual_hash != record.sha256 {
            return Err(format!(
                "hash mismatch for {}: ledger {}, file {}",
                record.logical_path, record.sha256, actual_hash
            ));
        }
        Ok((record.logical_path.clone(), actual_hash))
    }
}

fn validate_case(case: &CaseRecord, local_needs: bool) -> Result<(), String> {
    validate_sha256(&case.input_sha256, "input_sha256")?;
    validate_sha256(&case.tree_sha256, "tree_sha256")?;
    validate_file_records_sorted(&case.input_files, "input_files")?;
    validate_file_records_sorted(&case.files, "files")?;
    if case.origin.origin_path.is_empty() {
        return Err(format!("empty origin_path for {}", case.case_id));
    }
    validate_status_fields(case)?;

    if !local_needs
        && (case.needs_json.is_some()
            || case.needs_status.is_some()
            || case.needs_exit_code.is_some()
            || case.needs_warnings.is_some())
    {
        return Err(format!("core case {} has local-needs fields", case.case_id));
    }
    if let Some(record) = &case.needs_json {
        if record.storage != FileStorage::Ref {
            return Err(format!(
                "needs_json for {} must use ref storage",
                case.case_id
            ));
        }
        if !record.logical_path.ends_with("needs.json") {
            return Err(format!(
                "needs_json for {} has unexpected logical path",
                case.case_id
            ));
        }
    }
    if !local_needs && !case.input_files.is_empty() {
        // Core inputs are still valid; this branch documents that only the
        // four needs fields, not input materialization, are profile-specific.
    }
    Ok(())
}

fn validate_status_fields(case: &CaseRecord) -> Result<(), String> {
    match case.status {
        CaseStatus::Built if case.exit_code != Some(0) => {
            Err(format!("built case {} must have exit_code 0", case.case_id))
        }
        CaseStatus::BuildError if case.exit_code.is_none() || case.exit_code == Some(0) => {
            Err(format!(
                "build-error case {} must have a nonzero exit_code",
                case.case_id
            ))
        }
        status if status.is_excluded() => {
            if case.exit_code.is_some() {
                return Err(format!("excluded case {} has an exit_code", case.case_id));
            }
            if case.excluded_reason.is_none() {
                return Err(format!("excluded case {} has no reason", case.case_id));
            }
            if !case.warnings.is_empty() || !case.input_files.is_empty() || !case.files.is_empty() {
                return Err(format!(
                    "excluded case {} has captured artifacts",
                    case.case_id
                ));
            }
            Ok(())
        }
        _ => {
            if case.status.is_runnable() && case.excluded_reason.is_some() {
                return Err(format!(
                    "runnable case {} has an excluded reason",
                    case.case_id
                ));
            }
            Ok(())
        }
    }
}

fn validate_file_records_sorted(records: &[FileRecord], field: &str) -> Result<(), String> {
    let mut previous: Option<&str> = None;
    let mut logical_paths = BTreeSet::new();
    for record in records {
        validate_relative_path(&record.logical_path, &format!("{field}.logical_path"))?;
        validate_relative_path(&record.storage_path, &format!("{field}.storage_path"))?;
        if !logical_paths.insert(record.logical_path.as_str()) {
            return Err(format!(
                "duplicate logical path in {field}: {}",
                record.logical_path
            ));
        }
        if let Some(previous) = previous {
            if record.logical_path.as_str() < previous {
                return Err(format!("{field} are not sorted"));
            }
        }
        previous = Some(&record.logical_path);
    }
    Ok(())
}

fn validate_relative_path(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty()
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.contains('\\')
        || (value.len() >= 2 && value.as_bytes()[1] == b':')
    {
        return Err(format!("{field} is not a relative slash path: {value:?}"));
    }
    let path = Path::new(value);
    for component in path.components() {
        match component {
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(format!("{field} contains an escape: {value:?}"));
            }
            Component::Normal(_) => {}
        }
    }
    if value.split('/').any(|component| component.is_empty()) {
        return Err(format!("{field} contains an empty component: {value:?}"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} is not a SHA-256 hex digest: {value:?}"));
    }
    Ok(())
}

fn profile_name_is_local_needs(profile: &str) -> bool {
    profile == "local_needs"
}

fn reject_symlink_components(root: &Path, path: &Path) -> Result<(), String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| format!("path escapes root: {}", path.display()))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        if let Component::Normal(part) = component {
            current.push(part);
            if fs::symlink_metadata(&current)
                .map(|metadata| metadata.file_type().is_symlink())
                .unwrap_or(false)
            {
                return Err(format!("symlink in captured path: {}", current.display()));
            }
        }
    }
    Ok(())
}

fn validate_reverse_artifacts(
    profile_root: &Path,
    expected: &BTreeSet<String>,
) -> Result<(), String> {
    for directory in ["inputs", "refs", "blobs"] {
        let root = profile_root.join(directory);
        if !root.exists() {
            continue;
        }
        let mut stack = vec![root];
        while let Some(current) = stack.pop() {
            for entry in fs::read_dir(&current)
                .map_err(|error| format!("walk {}: {error}", current.display()))?
            {
                let entry = entry.map_err(|error| format!("read directory entry: {error}"))?;
                let path = entry.path();
                if entry
                    .file_type()
                    .map_err(|error| error.to_string())?
                    .is_dir()
                {
                    stack.push(path);
                } else if entry
                    .file_type()
                    .map_err(|error| error.to_string())?
                    .is_file()
                {
                    let relative = path
                        .strip_prefix(profile_root)
                        .map_err(|error| error.to_string())?
                        .to_string_lossy()
                        .replace(std::path::MAIN_SEPARATOR, "/");
                    if relative.ends_with("/warnings.txt") {
                        continue;
                    }
                    if !expected.contains(&relative) {
                        return Err(format!("unreferenced artifact {relative}"));
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn canonical_hash(entries: &[(String, String)]) -> String {
    let mut sorted = entries.to_vec();
    sorted.sort();
    let mut payload = String::new();
    for (path, digest) in sorted {
        payload.push_str(&path);
        payload.push('\0');
        payload.push_str(&digest);
        payload.push('\n');
    }
    sha256_hex(payload.as_bytes())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn profile_root(fixtures_root: &Path, profile: &str) -> PathBuf {
    fixtures_root.join(profile)
}
