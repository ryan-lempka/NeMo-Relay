// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process decision backend built on the libsy library.
//!
//! Instead of calling Switchyard's HTTP Decision API, the plugin drives a
//! libsy [`Algorithm`] in-process. The algorithm never performs a network
//! call: every model call it needs (including the routed call itself)
//! surfaces as a `CallLlm` promise that the plugin fulfills through Relay's
//! own dispatch chain, so provider credentials, retries, and observability
//! stay Relay-owned.

use std::collections::BTreeMap;
use std::sync::Arc;

use libsy::{Algorithm, LlmTarget, LlmTargetSet, RandomAlgo};
use libsy_examples::llm_class::LlmClassifierOrchAlgo;
use serde::{Deserialize, Serialize};

use crate::component::TargetBinding;

/// Which decision backend the plugin uses.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DecisionBackend {
    /// Call Switchyard's HTTP Decision API (requires a running server).
    #[default]
    Http,
    /// Make decisions in-process with an embedded libsy algorithm.
    Libsy,
}

impl DecisionBackend {
    /// Stable string form of the backend, used in events.
    pub fn label(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Libsy => "libsy",
        }
    }
}

/// The libsy reference algorithm to run.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum LibsyAlgorithmKind {
    /// Classify each request with a classifier model, then route strong/weak.
    #[default]
    LlmClassifier,
    /// Route uniformly at random among all configured targets.
    Random,
}

/// Configuration for the in-process libsy decision backend.
///
/// Target fields name entries in the plugin's `targets` map: libsy routes by
/// semantic name, and each semantic name is bound to a Relay-owned backend by
/// the existing `TargetBinding` table.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LibsyBackendConfig {
    /// The algorithm to run.
    #[serde(default)]
    pub algorithm: LibsyAlgorithmKind,
    /// Target that scores each request (llm_classifier only).
    #[serde(default)]
    pub classifier_target: String,
    /// Target routed to at or above the threshold (llm_classifier only).
    #[serde(default)]
    pub strong_target: String,
    /// Target routed to below the threshold (llm_classifier only).
    #[serde(default)]
    pub weak_target: String,
    /// Classifier score at or above which the strong target is chosen.
    #[serde(default = "default_threshold")]
    pub threshold: f64,
}

fn default_threshold() -> f64 {
    0.5
}

impl LibsyBackendConfig {
    /// The target binding IDs this algorithm can route to.
    pub(crate) fn routable_target_ids<'a>(
        &'a self,
        targets: &'a BTreeMap<String, TargetBinding>,
    ) -> Vec<&'a str> {
        match self.algorithm {
            LibsyAlgorithmKind::LlmClassifier => vec![
                self.classifier_target.as_str(),
                self.strong_target.as_str(),
                self.weak_target.as_str(),
            ],
            LibsyAlgorithmKind::Random => targets.keys().map(String::as_str).collect(),
        }
    }
}

/// Validate the libsy config against the plugin's target bindings.
pub(crate) fn validate_libsy_config(
    config: &LibsyBackendConfig,
    targets: &BTreeMap<String, TargetBinding>,
) -> Result<(), String> {
    match config.algorithm {
        LibsyAlgorithmKind::LlmClassifier => {
            for (field, id) in [
                ("classifier_target", &config.classifier_target),
                ("strong_target", &config.strong_target),
                ("weak_target", &config.weak_target),
            ] {
                if id.is_empty() {
                    return Err(format!("libsy llm_classifier requires {field}"));
                }
                if !targets.contains_key(id) {
                    return Err(format!("libsy {field} {id:?} has no target binding"));
                }
            }
            if !(0.0..=1.0).contains(&config.threshold) {
                return Err("libsy threshold must be within [0, 1]".into());
            }
        }
        LibsyAlgorithmKind::Random => {
            if targets.is_empty() {
                return Err("libsy random requires at least one target binding".into());
            }
        }
    }
    Ok(())
}

/// Build the configured algorithm over client-less targets.
///
/// Targets carry no `LlmClient`, so every model call the algorithm makes is
/// offloaded as a `CallLlm` promise for the plugin to fulfill.
pub(crate) fn build_algorithm(
    config: &LibsyBackendConfig,
    targets: &BTreeMap<String, TargetBinding>,
) -> Arc<dyn Algorithm> {
    match config.algorithm {
        LibsyAlgorithmKind::LlmClassifier => {
            let set = client_less_targets([
                config.classifier_target.as_str(),
                config.strong_target.as_str(),
                config.weak_target.as_str(),
            ]);
            Arc::new(LlmClassifierOrchAlgo::new(
                &config.classifier_target,
                &config.strong_target,
                &config.weak_target,
                config.threshold,
                set,
            ))
        }
        LibsyAlgorithmKind::Random => {
            let set = client_less_targets(targets.keys().map(String::as_str));
            Arc::new(RandomAlgo::new(set))
        }
    }
}

fn client_less_targets<'a>(names: impl IntoIterator<Item = &'a str>) -> LlmTargetSet {
    LlmTargetSet::new(
        names
            .into_iter()
            .map(|name| LlmTarget {
                semantic_name: name.to_string(),
                llm_client: None,
            })
            .collect(),
    )
}
