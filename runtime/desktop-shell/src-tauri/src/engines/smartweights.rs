use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SUPPORTED_CHUNK_ID: &str = "imagegen_tool_router_v1";
const DEFAULT_FEATURE_DIM: usize = 512;
const DEFAULT_EPOCHS: usize = 24;
const DEFAULT_LEARNING_RATE: f32 = 0.12;
const IMAGE_ROUTE_THRESHOLD: f32 = 0.65;

#[derive(Debug, Clone, Deserialize)]
pub struct SmartWeightsTrainMicrochunkRequest {
    pub chunk_id: Option<String>,
    pub corpus_paths: Option<Vec<String>>,
    pub feature_dim: Option<usize>,
    pub epochs: Option<usize>,
    pub learning_rate: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmartWeightsTrainMicrochunkResult {
    pub success: bool,
    pub chunk_id: String,
    pub chunk_kind: String,
    pub examples: usize,
    pub positives: usize,
    pub negatives: usize,
    pub feature_dim: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub trainable_overlay_parameters: usize,
    pub accuracy: f32,
    pub avg_loss: f32,
    pub manifest_path: String,
    pub weights_path: String,
    pub report_path: String,
    pub cleanup_removed: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmartWeightsImageRouteDecision {
    pub available: bool,
    pub chunk_id: String,
    pub score: f32,
    pub threshold: f32,
    pub use_image_turn_prompt: bool,
    pub reason: String,
}

#[derive(Debug, Clone)]
struct ChunkSpec {
    id: &'static str,
    pack_id: &'static str,
    pack_name: &'static str,
    role: &'static str,
    capability: &'static str,
}

#[derive(Debug, Clone)]
struct Sample {
    features: Vec<(usize, f32)>,
    label: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct JsonlTextRow {
    text: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolHeadFeatureSpace {
    feature_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolHeadArtifact {
    chunk_id: String,
    feature_space: ToolHeadFeatureSpace,
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Debug, Clone, Serialize)]
struct MetricsReport {
    chunk_id: String,
    chunk_kind: String,
    examples: usize,
    positives: usize,
    negatives: usize,
    feature_dim: usize,
    epochs: usize,
    learning_rate: f32,
    trainable_overlay_parameters: usize,
    accuracy: f32,
    avg_loss: f32,
    corpora: Vec<String>,
}

pub fn train_microchunk(
    repo_root: &Path,
    req: SmartWeightsTrainMicrochunkRequest,
) -> Result<SmartWeightsTrainMicrochunkResult, String> {
    let chunk_id = req
        .chunk_id
        .clone()
        .unwrap_or_else(|| SUPPORTED_CHUNK_ID.to_string());
    let spec = resolve_chunk_spec(&chunk_id)?;

    let feature_dim = req
        .feature_dim
        .unwrap_or(DEFAULT_FEATURE_DIM)
        .clamp(64, 4096);
    let epochs = req.epochs.unwrap_or(DEFAULT_EPOCHS).clamp(1, 256);
    let learning_rate = req
        .learning_rate
        .unwrap_or(DEFAULT_LEARNING_RATE)
        .clamp(0.0001, 2.0);

    let corpora = resolve_corpora(repo_root, req.corpus_paths)?;
    let temp_root = repo_root.join("smartweights").join("tmp");
    let run_id = unique_run_id(spec.id);
    let temp_dir = temp_root.join(&run_id);
    fs::create_dir_all(&temp_dir)
        .map_err(|e| format!("failed to create temp dir {}: {e}", temp_dir.display()))?;

    let train_result = (|| {
        let samples = load_samples(&corpora, feature_dim)?;
        if samples.is_empty() {
            return Err("no trainable samples found in the selected corpora".to_string());
        }

        let positives = samples.iter().filter(|sample| sample.label).count();
        let negatives = samples.len().saturating_sub(positives);
        if positives == 0 {
            return Err("microchunk corpus has no positive image-routing examples".to_string());
        }
        if negatives == 0 {
            return Err("microchunk corpus has no negative examples; add at least one normal chat row".to_string());
        }

        let (weights, bias) = train_logistic_head(&samples, feature_dim, epochs, learning_rate);
        let (accuracy, avg_loss) = score_model(&samples, &weights, bias);

        let output_dir = repo_root
            .join("smartweights")
            .join("packs")
            .join(spec.pack_id)
            .join("microchunks")
            .join(spec.id);
        fs::create_dir_all(&output_dir)
            .map_err(|e| format!("failed to create output dir {}: {e}", output_dir.display()))?;

        let report_dir = repo_root.join("ai").join("training").join("reports");
        fs::create_dir_all(&report_dir)
            .map_err(|e| format!("failed to create report dir {}: {e}", report_dir.display()))?;

        let weights_path = output_dir.join("weights.json");
        let manifest_path = output_dir.join("smartweight.json");
        let report_path = report_dir.join(format!("eldrae_{}_train.json", spec.id));

        let weights_tmp = temp_dir.join("weights.json.tmp");
        let manifest_tmp = temp_dir.join("smartweight.json.tmp");
        let report_tmp = temp_dir.join("report.json.tmp");

        let rel_corpora: Vec<String> = corpora
            .iter()
            .map(|path| relative_to_root(repo_root, path))
            .collect();

        let report = MetricsReport {
            chunk_id: spec.id.to_string(),
            chunk_kind: "tool_head".to_string(),
            examples: samples.len(),
            positives,
            negatives,
            feature_dim,
            epochs,
            learning_rate,
            trainable_overlay_parameters: feature_dim + 1,
            accuracy,
            avg_loss,
            corpora: rel_corpora.clone(),
        };

        let weights_json = json!({
            "schema_version": "smartweights/tool_head/v1",
            "chunk_id": spec.id,
            "kind": "binary_tool_router",
            "labels": ["default_chat", "imagegen_call"],
            "feature_space": {
                "type": "regex_word_hash_v1",
                "feature_dim": feature_dim,
                "includes": ["system", "user", "unigram", "bigram"]
            },
            "weights": weights,
            "bias": bias,
            "trained_at": unix_timestamp_string(),
        });
        fs::write(
            &weights_tmp,
            serde_json::to_vec_pretty(&weights_json)
                .map_err(|e| format!("failed to encode weights JSON: {e}"))?,
        )
        .map_err(|e| format!("failed to write {}: {e}", weights_tmp.display()))?;

        fs::write(
            &report_tmp,
            serde_json::to_vec_pretty(&report)
                .map_err(|e| format!("failed to encode report JSON: {e}"))?,
        )
        .map_err(|e| format!("failed to write {}: {e}", report_tmp.display()))?;

        let weights_hash = sha256_hex(&weights_tmp)?;
        let weights_bytes = file_len(&weights_tmp)?;
        let report_hash = sha256_hex(&report_tmp)?;
        let report_bytes = file_len(&report_tmp)?;

        let manifest_json = json!({
            "schema_version": "smartweights/v1",
            "name": spec.pack_name,
            "version": "0.1.0",
            "kind": "tool_head",
            "created_at": unix_timestamp_string(),
            "identity": {
                "assistant": "Eldrae",
                "role": spec.role
            },
            "base": {
                "runtime_model": "qwen2.5-coder:3b",
                "training_model": "rust.smartweights.microchunk",
                "frozen": true
            },
            "rft_space": {
                "family": "RFTMW",
                "encoding": "tool_head_hash_logreg",
                "base_tensor_mutation": "none"
            },
            "artifacts": [
                {
                    "path": relative_to_root(repo_root, &weights_path),
                    "optional": false,
                    "present": true,
                    "kind": "tool_head_weights",
                    "sha256": weights_hash,
                    "bytes": weights_bytes,
                    "file_count": 1
                },
                {
                    "path": relative_to_root(repo_root, &report_path),
                    "optional": false,
                    "present": true,
                    "kind": "training_report",
                    "sha256": report_hash,
                    "bytes": report_bytes,
                    "file_count": 1
                }
            ],
            "training": {
                "datasets": rel_corpora,
                "reports": [relative_to_root(repo_root, &report_path)],
                "guardrails": "Rust-only microchunk trainer; overlay-only routing head; no base weight mutation"
            },
            "capabilities": [
                "chat",
                "tool_planning",
                spec.capability
            ],
            "efficiency": {
                "goal": "tiny_rust_only_overlay_for_local_tool_routing",
                "budget": {
                    "model": "qwen2.5-coder:3b",
                    "base_parameters": 3000000000u64,
                    "active_dense_parameters": 3000000000u64,
                    "trainable_overlay_parameters": feature_dim + 1,
                    "actual_trainable_overlay_parameters": feature_dim + 1,
                    "parameters_per_layer": 0,
                    "storage_estimate": {
                        "dtype_bytes": 4,
                        "fp_bytes": weights_bytes,
                        "fp_mb": ((weights_bytes as f64) / (1024.0 * 1024.0)),
                        "q8_mb": ((weights_bytes as f64) / (1024.0 * 1024.0 * 2.0)),
                        "q4_mb": ((weights_bytes as f64) / (1024.0 * 1024.0 * 4.0))
                    }
                }
            },
            "safety": {
                "base_weight_mutation": "forbidden",
                "loads_as_overlay": true,
                "requires_manifest_validation": true,
                "destructive_actions": "requires_user_approval",
                "network_access": "disabled",
                "allowed_write_roots": [
                    "smartweights/packs",
                    "ai/training/reports",
                    "smartweights/tmp"
                ]
            },
            "notes": "Rust-trained binary tool router microchunk. This is a small SmartWeights head for deciding when Eldrae should emit a local [IMAGEGEN] call. It does not mutate base model tensors and does not use Python."
        });
        fs::write(
            &manifest_tmp,
            serde_json::to_vec_pretty(&manifest_json)
                .map_err(|e| format!("failed to encode manifest JSON: {e}"))?,
        )
        .map_err(|e| format!("failed to write {}: {e}", manifest_tmp.display()))?;

        commit_file(&weights_tmp, &weights_path)?;
        commit_file(&report_tmp, &report_path)?;
        commit_file(&manifest_tmp, &manifest_path)?;

        Ok(SmartWeightsTrainMicrochunkResult {
            success: true,
            chunk_id: spec.id.to_string(),
            chunk_kind: "tool_head".to_string(),
            examples: samples.len(),
            positives,
            negatives,
            feature_dim,
            epochs,
            learning_rate,
            trainable_overlay_parameters: feature_dim + 1,
            accuracy,
            avg_loss,
            manifest_path: manifest_path.to_string_lossy().to_string(),
            weights_path: weights_path.to_string_lossy().to_string(),
            report_path: report_path.to_string_lossy().to_string(),
            cleanup_removed: 0,
        })
    })();

    let cleanup_removed = cleanup_after_run(repo_root, &temp_dir);

    match train_result {
        Ok(mut result) => {
            result.cleanup_removed = cleanup_removed;
            Ok(result)
        }
        Err(error) => Err(format!("{error} [cleanup removed {cleanup_removed} temp entries]")),
    }
}

pub fn route_image_request(
    repo_root: &Path,
    messages: &[(String, String)],
) -> SmartWeightsImageRouteDecision {
    let Some(last_user) = messages
        .iter()
        .rev()
        .find(|(role, _)| role.eq_ignore_ascii_case("user"))
        .map(|(_, content)| content.trim())
    else {
        return SmartWeightsImageRouteDecision {
            available: false,
            chunk_id: SUPPORTED_CHUNK_ID.to_string(),
            score: 0.0,
            threshold: IMAGE_ROUTE_THRESHOLD,
            use_image_turn_prompt: false,
            reason: "no user message in chat history".to_string(),
        };
    };

    if last_user.is_empty() {
        return SmartWeightsImageRouteDecision {
            available: false,
            chunk_id: SUPPORTED_CHUNK_ID.to_string(),
            score: 0.0,
            threshold: IMAGE_ROUTE_THRESHOLD,
            use_image_turn_prompt: false,
            reason: "last user message is empty".to_string(),
        };
    }

    let artifact = match load_tool_head_artifact(repo_root) {
        Ok(artifact) => artifact,
        Err(reason) => {
            return SmartWeightsImageRouteDecision {
                available: false,
                chunk_id: SUPPORTED_CHUNK_ID.to_string(),
                score: 0.0,
                threshold: IMAGE_ROUTE_THRESHOLD,
                use_image_turn_prompt: false,
                reason,
            };
        }
    };

    let system_context = messages
        .iter()
        .filter(|(role, _)| role.eq_ignore_ascii_case("system"))
        .map(|(_, content)| content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let token_re = match Regex::new(r"[A-Za-z0-9_]+") {
        Ok(regex) => regex,
        Err(error) => {
            return SmartWeightsImageRouteDecision {
                available: false,
                chunk_id: artifact.chunk_id,
                score: 0.0,
                threshold: IMAGE_ROUTE_THRESHOLD,
                use_image_turn_prompt: false,
                reason: format!("router tokenizer regex error: {error}"),
            };
        }
    };

    let source = format!("{system_context}\n{last_user}");
    let features = match featurize_source_text(&source, artifact.feature_space.feature_dim, &token_re) {
        Some(features) => features,
        None => {
            return SmartWeightsImageRouteDecision {
                available: true,
                chunk_id: artifact.chunk_id,
                score: 0.0,
                threshold: IMAGE_ROUTE_THRESHOLD,
                use_image_turn_prompt: false,
                reason: "router could not featurize the current turn".to_string(),
            };
        }
    };

    let score = sigmoid(artifact.bias + dot(&artifact.weights, &features));
    let use_image_turn_prompt = score >= IMAGE_ROUTE_THRESHOLD;
    let reason = if use_image_turn_prompt {
        format!("trained tool-head matched image-routing pattern (score {:.3})", score)
    } else {
        format!("trained tool-head kept normal chat route (score {:.3})", score)
    };

    SmartWeightsImageRouteDecision {
        available: true,
        chunk_id: artifact.chunk_id,
        score,
        threshold: IMAGE_ROUTE_THRESHOLD,
        use_image_turn_prompt,
        reason,
    }
}

fn resolve_chunk_spec(chunk_id: &str) -> Result<ChunkSpec, String> {
    if chunk_id == SUPPORTED_CHUNK_ID {
        Ok(ChunkSpec {
            id: SUPPORTED_CHUNK_ID,
            pack_id: "eldrae-tool-router",
            pack_name: "Eldrae Imagegen Tool Router Microchunk",
            role: "local image generation routing specialist",
            capability: "visual_programs",
        })
    } else {
        Err(format!(
            "unsupported microchunk '{chunk_id}'. Rust trainer currently supports only '{SUPPORTED_CHUNK_ID}'"
        ))
    }
}

fn resolve_corpora(
    repo_root: &Path,
    requested: Option<Vec<String>>,
) -> Result<Vec<PathBuf>, String> {
    let rel_paths = requested.unwrap_or_else(|| {
        vec!["smartweights/training/eldrae_runtime_growth.jsonl".to_string()]
    });
    let mut corpora = Vec::new();
    for rel in rel_paths {
        let path = repo_root.join(&rel);
        if !path.exists() {
            return Err(format!("corpus not found: {}", path.display()));
        }
        corpora.push(path);
    }
    Ok(corpora)
}

fn load_tool_head_artifact(repo_root: &Path) -> Result<ToolHeadArtifact, String> {
    let path = repo_root
        .join("smartweights")
        .join("packs")
        .join("eldrae-tool-router")
        .join("microchunks")
        .join(SUPPORTED_CHUNK_ID)
        .join("weights.json");
    if !path.exists() {
        return Err(format!("trained microchunk head missing at {}", path.display()));
    }
    let raw = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read microchunk head {}: {e}", path.display()))?;
    let artifact: ToolHeadArtifact = serde_json::from_str(&raw)
        .map_err(|e| format!("failed to parse microchunk head {}: {e}", path.display()))?;
    if artifact.weights.len() != artifact.feature_space.feature_dim {
        return Err(format!(
            "microchunk head width mismatch: weights={} feature_dim={}",
            artifact.weights.len(),
            artifact.feature_space.feature_dim
        ));
    }
    Ok(artifact)
}

fn load_samples(corpora: &[PathBuf], feature_dim: usize) -> Result<Vec<Sample>, String> {
    let token_re = Regex::new(r"[A-Za-z0-9_]+").map_err(|e| format!("tokenizer regex error: {e}"))?;
    let mut samples = Vec::new();
    for corpus_path in corpora {
        let raw = fs::read_to_string(corpus_path)
            .map_err(|e| format!("failed to read corpus {}: {e}", corpus_path.display()))?;
        for (index, line) in raw.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let row: JsonlTextRow = serde_json::from_str(trimmed).map_err(|e| {
                format!(
                    "invalid JSONL row in {} at line {}: {e}",
                    corpus_path.display(),
                    index + 1
                )
            })?;
            if let Some(sample) = sample_from_text(&row.text, feature_dim, &token_re) {
                samples.push(sample);
            }
        }
    }
    Ok(samples)
}

fn sample_from_text(text: &str, feature_dim: usize, token_re: &Regex) -> Option<Sample> {
    let system = extract_segment(text, "system");
    let user = extract_segment(text, "user");
    let assistant = extract_segment(text, "assistant");
    if user.is_empty() || assistant.is_empty() {
        return None;
    }

    let label = assistant.contains("[IMAGEGEN]");
    let source = format!("{system}\n{user}");
    let features = featurize_source_text(&source, feature_dim, token_re)?;

    Some(Sample { features, label })
}

fn featurize_source_text(
    source: &str,
    feature_dim: usize,
    token_re: &Regex,
) -> Option<Vec<(usize, f32)>> {
    let mut counts: HashMap<usize, f32> = HashMap::new();
    let lowered = source.to_lowercase();
    let tokens: Vec<&str> = token_re.find_iter(&lowered).map(|m| m.as_str()).collect();
    if tokens.is_empty() {
        return None;
    }

    for token in &tokens {
        let idx = stable_hash(token.as_bytes()) as usize % feature_dim;
        *counts.entry(idx).or_insert(0.0) += 1.0;
    }
    for window in tokens.windows(2) {
        if let [left, right] = window {
            let mut bigram = String::with_capacity(left.len() + right.len() + 1);
            bigram.push_str(left);
            bigram.push(':');
            bigram.push_str(right);
            let idx = stable_hash(bigram.as_bytes()) as usize % feature_dim;
            *counts.entry(idx).or_insert(0.0) += 1.0;
        }
    }

    let norm = counts.values().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
    let mut features: Vec<(usize, f32)> = counts
        .into_iter()
        .map(|(idx, value)| (idx, value / norm))
        .collect();
    features.sort_by_key(|(idx, _)| *idx);
    Some(features)
}

fn train_logistic_head(
    samples: &[Sample],
    feature_dim: usize,
    epochs: usize,
    learning_rate: f32,
) -> (Vec<f32>, f32) {
    let mut weights = vec![0.0f32; feature_dim];
    let mut bias = 0.0f32;
    let positives = samples.iter().filter(|sample| sample.label).count().max(1);
    let negatives = samples.len().saturating_sub(positives).max(1);
    let positive_weight = negatives as f32 / positives as f32;

    for _ in 0..epochs {
        for sample in samples {
            let logit = bias + dot(&weights, &sample.features);
            let prediction = sigmoid(logit);
            let target = if sample.label { 1.0 } else { 0.0 };
            let sample_weight = if sample.label { positive_weight } else { 1.0 };
            let gradient = (prediction - target) * sample_weight;

            bias -= learning_rate * gradient;
            for (idx, value) in &sample.features {
                weights[*idx] -= learning_rate * gradient * *value;
            }
        }
    }

    (weights, bias)
}

fn score_model(samples: &[Sample], weights: &[f32], bias: f32) -> (f32, f32) {
    let mut correct = 0usize;
    let mut loss_sum = 0.0f32;
    for sample in samples {
        let logit = bias + dot(weights, &sample.features);
        let prediction = sigmoid(logit).clamp(1e-6, 1.0 - 1e-6);
        let target = if sample.label { 1.0 } else { 0.0 };
        let predicted_label = prediction >= 0.5;
        if predicted_label == sample.label {
            correct += 1;
        }
        loss_sum += -(target * prediction.ln() + (1.0 - target) * (1.0 - prediction).ln());
    }

    (
        correct as f32 / samples.len().max(1) as f32,
        loss_sum / samples.len().max(1) as f32,
    )
}

fn dot(weights: &[f32], features: &[(usize, f32)]) -> f32 {
    let mut total = 0.0f32;
    for (idx, value) in features {
        total += weights[*idx] * *value;
    }
    total
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

fn extract_segment(text: &str, role: &str) -> String {
    let marker = format!("<|im_start|>{role}\n");
    if let Some(start) = text.find(&marker) {
        let rest = &text[start + marker.len()..];
        if let Some(end) = rest.find("<|im_end|>") {
            return rest[..end].trim().to_string();
        }
    }
    String::new()
}

fn commit_file(src: &Path, dest: &Path) -> Result<(), String> {
    if dest.exists() {
        fs::remove_file(dest)
            .map_err(|e| format!("failed to replace {}: {e}", dest.display()))?;
    }
    fs::rename(src, dest)
        .map_err(|e| format!("failed to finalize {}: {e}", dest.display()))
}

fn cleanup_after_run(repo_root: &Path, temp_dir: &Path) -> usize {
    let mut removed = 0usize;

    if temp_dir.exists() {
        removed += remove_path_count(temp_dir);
    }

    let temp_root = repo_root.join("smartweights").join("tmp");
    if let Ok(entries) = fs::read_dir(&temp_root) {
        let prefix = format!("microchunk_{}_", SUPPORTED_CHUNK_ID);
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) {
                removed += remove_path_count(&path);
            }
        }
    }

    let output_dir = repo_root
        .join("smartweights")
        .join("packs")
        .join("eldrae-tool-router")
        .join("microchunks")
        .join(SUPPORTED_CHUNK_ID);
    if let Ok(entries) = fs::read_dir(&output_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("tmp"))
                .unwrap_or(false)
            {
                removed += remove_path_count(&path);
            }
        }
    }

    removed
}

fn remove_path_count(path: &Path) -> usize {
    let count = count_entries(path);
    let _ = if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    count
}

fn count_entries(path: &Path) -> usize {
    if path.is_file() {
        return 1;
    }
    let mut total = 1usize;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            total += count_entries(&entry.path());
        }
    }
    total
}

fn sha256_hex(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to hash {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn file_len(path: &Path) -> Result<u64, String> {
    fs::metadata(path)
        .map(|meta| meta.len())
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))
}

fn relative_to_root(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn unix_timestamp_string() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}

fn unique_run_id(chunk_id: &str) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("microchunk_{chunk_id}_{millis}_{}", std::process::id())
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 1469598103934665603u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{
        extract_segment, load_samples, route_image_request, score_model, train_logistic_head,
        train_microchunk, SmartWeightsTrainMicrochunkRequest, SUPPORTED_CHUNK_ID,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn extracts_chat_segments() {
        let text = "<|im_start|>system\nhello<|im_end|>\n<|im_start|>user\nmake me a wallpaper<|im_end|>\n<|im_start|>assistant\n[IMAGEGEN]\nwidth=2048<|im_end|>";
        assert_eq!(extract_segment(text, "system"), "hello");
        assert_eq!(extract_segment(text, "user"), "make me a wallpaper");
        assert!(extract_segment(text, "assistant").contains("[IMAGEGEN]"));
    }

    #[test]
    fn tiny_training_run_separates_imagegen_rows() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("smartweights_test_{unique}.jsonl"));
        let body = [
            r#"{"text":"<|im_start|>system\nUse local tools.<|im_end|>\n<|im_start|>user\nmake me a 4k neon city wallpaper<|im_end|>\n<|im_start|>assistant\n[IMAGEGEN]\nprompt=\"neon city\"<|im_end|>"}"#,
            r#"{"text":"<|im_start|>system\nUse local tools.<|im_end|>\n<|im_start|>user\nhey<|im_end|>\n<|im_start|>assistant\nhey, i'm here.<|im_end|>"}"#,
        ]
        .join("\n");
        fs::write(&path, body).expect("write test corpus");

        let samples = load_samples(&[PathBuf::from(&path)], 128).expect("load samples");
        let (weights, bias) = train_logistic_head(&samples, 128, 40, 0.15);
        let (accuracy, _) = score_model(&samples, &weights, bias);

        let _ = fs::remove_file(&path);
        assert!(accuracy >= 0.99, "expected near-perfect fit, got {accuracy}");
    }

    #[test]
    fn end_to_end_microchunk_run_writes_artifacts_and_cleans_temp() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let repo_root = std::env::temp_dir().join(format!("smartweights_repo_{unique}"));
        let corpus_dir = repo_root.join("smartweights").join("training");
        fs::create_dir_all(&corpus_dir).expect("create corpus dir");

        let corpus_path = corpus_dir.join("eldrae_runtime_growth.jsonl");
        let body = [
            r#"{"text":"<|im_start|>system\nUse local tools precisely when needed.<|im_end|>\n<|im_start|>user\nMake me a 4K neon city wallpaper.<|im_end|>\n<|im_start|>assistant\nGenerating now.\n[IMAGEGEN]\nprompt=\"neon city\"<|im_end|>"}"#,
            r#"{"text":"<|im_start|>system\nKeep casual conversation short but complete.<|im_end|>\n<|im_start|>user\nhey<|im_end|>\n<|im_start|>assistant\nhey, i'm here.<|im_end|>"}"#,
        ]
        .join("\n");
        fs::write(&corpus_path, body).expect("write runtime growth corpus");

        let req = SmartWeightsTrainMicrochunkRequest {
            chunk_id: Some(SUPPORTED_CHUNK_ID.to_string()),
            corpus_paths: None,
            feature_dim: Some(128),
            epochs: Some(16),
            learning_rate: Some(0.15),
        };
        let result = train_microchunk(&repo_root, req).expect("train microchunk");

        assert!(PathBuf::from(&result.weights_path).exists());
        assert!(PathBuf::from(&result.manifest_path).exists());
        assert!(PathBuf::from(&result.report_path).exists());

        let temp_root = repo_root.join("smartweights").join("tmp");
        let leftover = fs::read_dir(&temp_root)
            .ok()
            .map(|entries| entries.count())
            .unwrap_or(0);
        assert_eq!(leftover, 0, "expected temp dir to be cleaned");

        let _ = fs::remove_dir_all(&repo_root);
    }

    #[test]
    fn trained_head_influences_live_route() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let repo_root = std::env::temp_dir().join(format!("smartweights_route_repo_{unique}"));
        let corpus_dir = repo_root.join("smartweights").join("training");
        fs::create_dir_all(&corpus_dir).expect("create corpus dir");

        let corpus_path = corpus_dir.join("eldrae_runtime_growth.jsonl");
        let body = [
            r#"{"text":"<|im_start|>system\nUse local tools precisely when needed.<|im_end|>\n<|im_start|>user\nMake me a 4K neon city wallpaper.<|im_end|>\n<|im_start|>assistant\nGenerating now.\n[IMAGEGEN]\nprompt=\"neon city\"<|im_end|>"}"#,
            r#"{"text":"<|im_start|>system\nKeep casual conversation short but complete.<|im_end|>\n<|im_start|>user\nhey<|im_end|>\n<|im_start|>assistant\nhey, i'm here.<|im_end|>"}"#,
        ]
        .join("\n");
        fs::write(&corpus_path, body).expect("write runtime growth corpus");

        let req = SmartWeightsTrainMicrochunkRequest {
            chunk_id: Some(SUPPORTED_CHUNK_ID.to_string()),
            corpus_paths: None,
            feature_dim: Some(128),
            epochs: Some(16),
            learning_rate: Some(0.15),
        };
        let _ = train_microchunk(&repo_root, req).expect("train microchunk");

        let image_route = route_image_request(
            &repo_root,
            &[
                ("system".to_string(), "Use local tools precisely when needed.".to_string()),
                ("user".to_string(), "Make me a 4K neon city wallpaper.".to_string()),
            ],
        );
        assert!(image_route.available);
        assert!(image_route.use_image_turn_prompt, "expected positive route, got {:?}", image_route);

        let chat_route = route_image_request(
            &repo_root,
            &[
                ("system".to_string(), "Keep casual conversation short but complete.".to_string()),
                ("user".to_string(), "hey".to_string()),
            ],
        );
        assert!(chat_route.available);
        assert!(
            !chat_route.use_image_turn_prompt,
            "expected normal chat route, got {:?}",
            chat_route
        );

        let _ = fs::remove_dir_all(&repo_root);
    }
}
