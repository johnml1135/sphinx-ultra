use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};

use flate2::read::ZlibDecoder;

pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Warnings,
    SearchIndex,
    NeedsJson,
    ObjectsInventory,
    TextCrlf,
    ExactBytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub category: String,
    pub logical_path: String,
    pub first_expected_line: Option<usize>,
    pub expected: String,
    pub actual: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct InventoryRecord {
    pub name: String,
    pub domain_role: String,
    pub priority: i32,
    pub uri: String,
    pub display_name: String,
}

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

pub fn policy_for_path(logical_path: &str) -> Policy {
    if logical_path == "searchindex.js" {
        Policy::SearchIndex
    } else if logical_path.rsplit('/').next() == Some("needs.json") {
        Policy::NeedsJson
    } else if logical_path == "objects.inv" {
        Policy::ObjectsInventory
    } else if logical_path.ends_with(".buildinfo") {
        Policy::ExactBytes
    } else if logical_path.starts_with("_sources/")
        || [".html", ".css", ".js", ".json", ".xml", ".txt"]
            .iter()
            .any(|suffix| logical_path.ends_with(suffix))
    {
        Policy::TextCrlf
    } else {
        Policy::ExactBytes
    }
}

pub fn compare_file(
    logical_path: &str,
    expected: &[u8],
    actual: &[u8],
    _source_root: Option<&Path>,
) -> Vec<Diagnostic> {
    match policy_for_path(logical_path) {
        Policy::SearchIndex => compare_json_file(
            logical_path,
            expected,
            actual,
            "searchindex-value",
            "invalid-searchindex",
            true,
        ),
        Policy::NeedsJson => compare_json_file(
            logical_path,
            expected,
            actual,
            "needs-json-value",
            "invalid-needs-json",
            false,
        ),
        Policy::ObjectsInventory => compare_inventory_file(logical_path, expected, actual),
        Policy::TextCrlf => {
            let expected = normalize_crlf(expected);
            let actual = normalize_crlf(actual);
            if expected == actual {
                Vec::new()
            } else {
                vec![value_diagnostic(
                    "text-value",
                    logical_path,
                    &expected,
                    &actual,
                )]
            }
        }
        Policy::Warnings | Policy::ExactBytes => {
            if expected == actual {
                Vec::new()
            } else {
                vec![value_diagnostic(
                    "bytes-value",
                    logical_path,
                    expected,
                    actual,
                )]
            }
        }
    }
}

pub fn compare_warnings(
    expected: &[u8],
    actual: &[u8],
    expected_source_root: Option<&Path>,
    actual_source_root: Option<&Path>,
) -> Vec<Diagnostic> {
    let expected = normalize_warning_bytes(expected, expected_source_root);
    let actual = normalize_warning_bytes(actual, actual_source_root);
    if expected == actual {
        Vec::new()
    } else {
        vec![value_diagnostic(
            "warning",
            "warnings",
            expected.as_bytes(),
            actual.as_bytes(),
        )]
    }
}

pub fn compare_trees(
    expected: &BTreeMap<String, Vec<u8>>,
    actual: &BTreeMap<String, Vec<u8>>,
    expected_warnings: Option<&[u8]>,
    actual_warnings: Option<&[u8]>,
    expected_status: Option<CaseStatus>,
    actual_status: Option<CaseStatus>,
) -> Vec<Diagnostic> {
    let mut paths = BTreeSet::new();
    paths.extend(expected.keys().cloned());
    paths.extend(actual.keys().cloned());
    let mut diagnostics = Vec::new();
    for path in paths {
        match (expected.get(&path), actual.get(&path)) {
            (None, Some(actual)) => diagnostics.push(Diagnostic {
                category: "unexpected-file".to_string(),
                logical_path: path,
                first_expected_line: None,
                expected: String::new(),
                actual: display_bytes(actual),
                detail: "file exists only in actual output".to_string(),
            }),
            (Some(expected), None) => diagnostics.push(Diagnostic {
                category: "missing-file".to_string(),
                logical_path: path,
                first_expected_line: Some(1),
                expected: display_bytes(expected),
                actual: String::new(),
                detail: "file exists only in expected output".to_string(),
            }),
            (Some(expected), Some(actual)) => {
                diagnostics.extend(compare_file(&path, expected, actual, None));
            }
            (None, None) => unreachable!(),
        }
    }
    if let (Some(expected), Some(actual)) = (expected_warnings, actual_warnings) {
        diagnostics.extend(compare_warnings(expected, actual, None, None));
    }
    if let (Some(expected), Some(actual)) = (expected_status, actual_status) {
        let expected_is_error = expected == CaseStatus::BuildError;
        let actual_is_error = actual == CaseStatus::BuildError;
        if expected_is_error != actual_is_error {
            diagnostics.push(Diagnostic {
                category: "status".to_string(),
                logical_path: String::new(),
                first_expected_line: None,
                expected: format_status(expected),
                actual: format_status(actual),
                detail: "build status class differs".to_string(),
            });
        }
    }
    diagnostics.sort_by(diagnostic_order);
    diagnostics
}

fn compare_json_file(
    logical_path: &str,
    expected: &[u8],
    actual: &[u8],
    value_category: &str,
    invalid_category: &str,
    searchindex_wrapper: bool,
) -> Vec<Diagnostic> {
    let expected_value = match parse_json_value(expected, searchindex_wrapper) {
        Ok(value) => value,
        Err(error) => {
            return vec![invalid_diagnostic(
                invalid_category,
                logical_path,
                format!("expected JSON is invalid: {error}"),
            )]
        }
    };
    let actual_value = match parse_json_value(actual, searchindex_wrapper) {
        Ok(value) => value,
        Err(error) => {
            return vec![invalid_diagnostic(
                invalid_category,
                logical_path,
                format!("actual JSON is invalid: {error}"),
            )]
        }
    };
    if expected_value == actual_value {
        Vec::new()
    } else {
        vec![value_diagnostic(
            value_category,
            logical_path,
            serde_json::to_string_pretty(&expected_value)
                .unwrap_or_default()
                .as_bytes(),
            serde_json::to_string_pretty(&actual_value)
                .unwrap_or_default()
                .as_bytes(),
        )]
    }
}

fn parse_json_value(bytes: &[u8], searchindex_wrapper: bool) -> Result<Value, String> {
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => normalize_crlf(text.as_bytes()),
        Err(error) => {
            return Err(format!("not UTF-8: {error}"));
        }
    };
    let text = String::from_utf8(text).map_err(|error| error.to_string())?;
    let payload = if searchindex_wrapper {
        let trimmed = text.trim_end();
        let prefix = "Search.setIndex(";
        if !trimmed.starts_with(prefix) || !trimmed.ends_with(");") {
            return Err("missing Search.setIndex(...) wrapper".to_string());
        }
        &trimmed[prefix.len()..trimmed.len() - 2]
    } else {
        text.as_str()
    };
    serde_json::from_str(payload).map_err(|error| error.to_string())
}

fn compare_inventory_file(logical_path: &str, expected: &[u8], actual: &[u8]) -> Vec<Diagnostic> {
    let expected_inventory = match parse_inventory(expected) {
        Ok(value) => value,
        Err(error) => {
            return vec![invalid_diagnostic(
                "invalid-objects-inventory",
                logical_path,
                format!("expected inventory is invalid: {error}"),
            )]
        }
    };
    let actual_inventory = match parse_inventory(actual) {
        Ok(value) => value,
        Err(error) => {
            return vec![invalid_diagnostic(
                "invalid-objects-inventory",
                logical_path,
                format!("actual inventory is invalid: {error}"),
            )]
        }
    };
    if expected_inventory == actual_inventory {
        Vec::new()
    } else {
        vec![value_diagnostic(
            "objects-inventory-value",
            logical_path,
            format_inventory(&expected_inventory).as_bytes(),
            format_inventory(&actual_inventory).as_bytes(),
        )]
    }
}

fn parse_inventory(bytes: &[u8]) -> Result<(Vec<u8>, Vec<InventoryRecord>), String> {
    const HEADER: &[u8] = b"# Sphinx inventory version 2\n# Project: ";
    if !bytes.starts_with(HEADER) {
        return Err("missing Sphinx inventory header".to_string());
    }
    let mut newline_positions = Vec::new();
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'\n' {
            newline_positions.push(index + 1);
            if newline_positions.len() == 4 {
                break;
            }
        }
    }
    if newline_positions.len() != 4 {
        return Err("inventory has fewer than four header lines".to_string());
    }
    let header_end = newline_positions[3];
    let header = bytes[..header_end].to_vec();
    let mut decoder = ZlibDecoder::new(Cursor::new(&bytes[header_end..]));
    let mut records_text = String::new();
    decoder
        .read_to_string(&mut records_text)
        .map_err(|error| format!("zlib decode failed: {error}"))?;
    let mut records = Vec::new();
    for (line_number, line) in records_text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields = line.splitn(5, ' ').collect::<Vec<_>>();
        if fields.len() != 5 || fields.iter().any(|field| field.is_empty()) {
            return Err(format!("invalid record on line {}", line_number + 1));
        }
        let priority = fields[2]
            .parse::<i32>()
            .map_err(|error| format!("invalid priority on line {}: {error}", line_number + 1))?;
        records.push(InventoryRecord {
            name: fields[0].to_string(),
            domain_role: fields[1].to_string(),
            priority,
            uri: fields[3].to_string(),
            display_name: fields[4].to_string(),
        });
    }
    records.sort();
    Ok((header, records))
}

fn format_inventory(inventory: &(Vec<u8>, Vec<InventoryRecord>)) -> String {
    let header = String::from_utf8_lossy(&inventory.0);
    let records = inventory
        .1
        .iter()
        .map(|record| {
            format!(
                "{} {} {} {} {}",
                record.name, record.domain_role, record.priority, record.uri, record.display_name
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("{header}{records}")
}

fn normalize_warning_bytes(bytes: &[u8], source_root: Option<&Path>) -> String {
    let mut value = String::from_utf8_lossy(&normalize_crlf(bytes)).into_owned();
    if let Some(source_root) = source_root {
        let root = source_root.to_string_lossy();
        value = value.replace(root.as_ref(), "<SRCDIR>");
    }
    value
}

fn normalize_crlf(bytes: &[u8]) -> Vec<u8> {
    let mut normalized = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
            normalized.push(b'\n');
            index += 2;
        } else {
            normalized.push(bytes[index]);
            index += 1;
        }
    }
    normalized
}

fn value_diagnostic(
    category: &str,
    logical_path: &str,
    expected: &[u8],
    actual: &[u8],
) -> Diagnostic {
    let expected = display_bytes(expected);
    let actual = display_bytes(actual);
    Diagnostic {
        category: category.to_string(),
        logical_path: logical_path.to_string(),
        first_expected_line: first_difference_line(&expected, &actual),
        expected,
        actual,
        detail: "normalized values differ".to_string(),
    }
}

fn invalid_diagnostic(category: &str, logical_path: &str, detail: String) -> Diagnostic {
    Diagnostic {
        category: category.to_string(),
        logical_path: logical_path.to_string(),
        first_expected_line: None,
        expected: String::new(),
        actual: String::new(),
        detail,
    }
}

fn display_bytes(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn first_difference_line(expected: &str, actual: &str) -> Option<usize> {
    let mut expected_lines = expected.lines();
    let mut actual_lines = actual.lines();
    let mut line_number = 1;
    loop {
        match (expected_lines.next(), actual_lines.next()) {
            (None, None) => return None,
            (left, right) if left == right => line_number += 1,
            _ => return Some(line_number),
        }
    }
}

fn format_status(status: CaseStatus) -> String {
    match status {
        CaseStatus::Built => "built",
        CaseStatus::BuildError => "build-error",
        CaseStatus::ReferenceCrash => "reference-crash",
        CaseStatus::ExcludedNetwork => "excluded-network",
        CaseStatus::ExcludedPlantuml => "excluded-plantuml",
    }
    .to_string()
}

fn diagnostic_order(left: &Diagnostic, right: &Diagnostic) -> std::cmp::Ordering {
    left.logical_path
        .cmp(&right.logical_path)
        .then_with(|| left.category.cmp(&right.category))
        .then_with(|| left.first_expected_line.cmp(&right.first_expected_line))
        .then_with(|| left.detail.cmp(&right.detail))
}
