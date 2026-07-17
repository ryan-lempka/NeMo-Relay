// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::error::{FlowError, Result};
use serde_json::{Map, Value as Json};
use switchyard_translation::{
    AggLlmResponse, ContentBlock, DeterministicIdPolicy, ImageSource,
    LlmRequest as TranslationRequest, LlmResponseChunk, LossyConversionPolicy, PreservationPolicy,
    Role, StreamCodec, StreamCodecRegistry, StreamTranslationState, TargetCapabilities,
    TranslationDiagnostic, TranslationEngine, TranslationPolicy, UnknownFieldPolicy, WireFormat,
};

use crate::component::WireProtocol;

pub(crate) fn translation_engine() -> TranslationEngine {
    TranslationEngine::default()
}

pub(crate) fn decode_request(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    request: &LlmRequest,
) -> Result<TranslationRequest> {
    let output = engine
        .decode_request(
            wire_format(protocol),
            &request.content,
            &translation_policy(),
        )
        .map_err(translation_error)?;
    ensure_no_diagnostics(&output.diagnostics)?;
    Ok(output.request)
}

pub(crate) fn encode_request(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    request: &TranslationRequest,
    headers: Map<String, Json>,
) -> Result<LlmRequest> {
    let output = engine
        .encode_request(wire_format(protocol), request, &translation_policy())
        .map_err(translation_error)?;
    ensure_no_diagnostics(&output.diagnostics)?;
    Ok(LlmRequest {
        headers,
        content: output.body,
    })
}

/// Lenient policy for decoding a request the plugin will not re-encode
/// losslessly: unknown provider fields are preserved instead of rejected, so
/// bodies carrying extensions like `cache_control` still yield an IR the
/// routing algorithm can read. Only safe when dispatch is same-protocol (the
/// wire body is passed through raw, not re-encoded from this IR).
fn lenient_policy() -> TranslationPolicy {
    TranslationPolicy {
        unknown_field_policy: UnknownFieldPolicy::Preserve,
        lossy_conversion_policy: LossyConversionPolicy::AllowWithDiagnostics,
        ..translation_policy()
    }
}

pub(crate) fn decode_request_lenient(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    request: &LlmRequest,
) -> Result<TranslationRequest> {
    let output = engine
        .decode_request(wire_format(protocol), &request.content, &lenient_policy())
        .map_err(translation_error)?;
    Ok(output.request)
}

pub(crate) fn decode_response_lenient(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    body: &Json,
) -> Result<AggLlmResponse> {
    let output = engine
        .decode_response(wire_format(protocol), body, &lenient_policy())
        .map_err(translation_error)?;
    Ok(output.response)
}

pub(crate) fn decode_response(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    body: &Json,
) -> Result<AggLlmResponse> {
    let output = engine
        .decode_response(wire_format(protocol), body, &translation_policy())
        .map_err(translation_error)?;
    ensure_no_diagnostics(&output.diagnostics)?;
    Ok(output.response)
}

pub(crate) fn encode_response(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    response: &AggLlmResponse,
) -> Result<Json> {
    let output = engine
        .encode_response(wire_format(protocol), response, &translation_policy())
        .map_err(translation_error)?;
    ensure_no_diagnostics(&output.diagnostics)?;
    Ok(output.body)
}

/// Decodes one provider's wire stream events into neutral response chunks.
///
/// Wraps one stream codec plus its per-stream state; feed it every event of
/// a single provider stream in order.
pub(crate) struct ChunkDecoder {
    codec: Arc<dyn StreamCodec>,
    state: StreamTranslationState,
}

impl ChunkDecoder {
    pub(crate) fn new(protocol: WireProtocol) -> Result<Self> {
        let format = wire_format(protocol);
        let codec = StreamCodecRegistry::with_builtins()
            .codec(format)
            .map_err(translation_error)?;
        Ok(Self {
            codec,
            state: StreamTranslationState::new(format, format),
        })
    }

    /// Decodes one wire stream event into zero or more neutral chunks.
    pub(crate) fn decode(&mut self, event: &Json) -> Vec<LlmResponseChunk> {
        self.codec.decode_event(&mut self.state, event)
    }
}

/// Encodes neutral response chunks back into one provider's wire stream events.
///
/// Wraps one stream codec plus its per-stream state; feed it every chunk of a
/// single response stream in order, then call [`finish`](Self::finish).
pub(crate) struct ChunkEncoder {
    codec: Arc<dyn StreamCodec>,
    state: StreamTranslationState,
}

impl ChunkEncoder {
    pub(crate) fn new(protocol: WireProtocol) -> Result<Self> {
        let format = wire_format(protocol);
        let codec = StreamCodecRegistry::with_builtins()
            .codec(format)
            .map_err(translation_error)?;
        Ok(Self {
            codec,
            state: StreamTranslationState::new(format, format),
        })
    }

    /// Encodes one neutral chunk into zero or more wire stream events.
    pub(crate) fn encode(&mut self, chunk: LlmResponseChunk) -> Vec<Json> {
        self.codec.encode_event(&mut self.state, chunk)
    }

    /// Emits any terminal wire events needed after the chunk stream ends.
    pub(crate) fn finish(&mut self) -> Vec<Json> {
        self.codec.finish(&mut self.state)
    }
}

pub(crate) fn validate_portable_request(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    request: &LlmRequest,
) -> Result<()> {
    const RESTRICTED_KEYS: &[&str] = &[
        "cache_control",
        "audio",
        "thinking",
        "computer_use",
        "server_tool_use",
    ];
    if protocol == WireProtocol::AnthropicMessages
        && contains_invalid_anthropic_image_source(&request.content)
    {
        return Err(FlowError::InvalidArgument(
            "request uses an unsupported or malformed Anthropic image source".into(),
        ));
    }
    if contains_any_key_recursive(&request.content, RESTRICTED_KEYS) {
        return Err(FlowError::InvalidArgument(
            "request uses a provider-specific extension that requires same-protocol fail-open"
                .into(),
        ));
    }
    let decoded = decode_request(engine, protocol, request)?;
    if decoded.reasoning.effort.is_some()
        || decoded.reasoning.raw.is_some()
        || decoded
            .extensions
            .fields
            .iter()
            .any(|(key, value)| key != "stream_options" || !portable_stream_options(value))
        || request_contains_unsupported_content(&decoded)
    {
        return Err(FlowError::InvalidArgument(
            "request uses provider-specific fields that cannot be translated safely".into(),
        ));
    }
    Ok(())
}

pub(crate) fn latest_user_prompt(request: &TranslationRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .and_then(|message| message.text_content("\n"))
}

pub(crate) fn recent_message_window(
    request: &TranslationRequest,
    count: usize,
) -> TranslationRequest {
    let mut window = request.clone();
    let split = window.messages.len().saturating_sub(count);
    window.messages = window.messages.split_off(split);
    window
}

pub(crate) fn translate_response(
    engine: &TranslationEngine,
    source: WireProtocol,
    target: WireProtocol,
    response: &Json,
) -> Result<Json> {
    if source == target {
        return Ok(response.clone());
    }
    ensure_portable_response(source, response)?;
    let output = engine
        .translate_response(
            wire_format(source),
            wire_format(target),
            response,
            &translation_policy(),
        )
        .map_err(translation_error)?;
    ensure_no_diagnostics(&output.diagnostics)?;
    Ok(output.body)
}

pub(crate) const fn wire_format(protocol: WireProtocol) -> WireFormat {
    match protocol {
        WireProtocol::OpenaiChat => WireFormat::OpenAiChat,
        WireProtocol::OpenaiResponses => WireFormat::OpenAiResponses,
        WireProtocol::AnthropicMessages => WireFormat::AnthropicMessages,
    }
}

fn translation_policy() -> TranslationPolicy {
    TranslationPolicy {
        unknown_field_policy: UnknownFieldPolicy::Reject,
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        deterministic_ids: DeterministicIdPolicy::GenerateStable {
            prefix: "relay".into(),
        },
        preservation: PreservationPolicy::Disabled,
        target_capabilities: TargetCapabilities::default(),
    }
}

fn translation_error(error: switchyard_translation::TranslationError) -> FlowError {
    FlowError::InvalidArgument(format!("Switchyard translation failed: {error}"))
}

fn ensure_no_diagnostics(diagnostics: &[TranslationDiagnostic]) -> Result<()> {
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(FlowError::InvalidArgument(format!(
            "Switchyard translation was not lossless: {diagnostics:?}"
        )))
    }
}

fn portable_stream_options(value: &Json) -> bool {
    let Some(options) = value.as_object() else {
        return false;
    };
    options.len() == 1 && options.get("include_usage").is_some_and(Json::is_boolean)
}

fn request_contains_unsupported_content(request: &TranslationRequest) -> bool {
    request
        .instructions
        .iter()
        .flat_map(|instruction| instruction.content.iter())
        .chain(
            request
                .messages
                .iter()
                .flat_map(|message| message.content.iter()),
        )
        .any(unsupported_content_block)
}

fn unsupported_content_block(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Text { .. } | ContentBlock::Refusal { .. } | ContentBlock::ToolCall(_) => {
            false
        }
        ContentBlock::Image { source } => invalid_image_source(source),
        ContentBlock::ToolResult(result) => result.content.iter().any(unsupported_content_block),
        ContentBlock::Reasoning { .. }
        | ContentBlock::Audio { .. }
        | ContentBlock::Video { .. }
        | ContentBlock::File { .. }
        | ContentBlock::Unknown { .. } => true,
    }
}

fn invalid_image_source(source: &ImageSource) -> bool {
    match source {
        ImageSource::Url { url, .. } => {
            url.starts_with("data:") && base64_data_uri_parts(url).is_none()
        }
        ImageSource::Base64 { media_type, data } => {
            media_type.as_deref().is_none_or(str::is_empty) || data.is_empty()
        }
        ImageSource::Raw(_) => true,
    }
}

fn base64_data_uri_parts(url: &str) -> Option<(&str, &str)> {
    let (metadata, data) = url.strip_prefix("data:")?.split_once(',')?;
    let media_type = metadata.strip_suffix(";base64")?;
    (!media_type.is_empty() && !data.is_empty()).then_some((media_type, data))
}

fn contains_invalid_anthropic_image_source(value: &Json) -> bool {
    match value {
        Json::Object(object) => {
            if object.get("type").and_then(Json::as_str) == Some("image") {
                let Some(source) = object.get("source").and_then(Json::as_object) else {
                    return true;
                };
                match source.get("type").and_then(Json::as_str) {
                    Some("url") => source
                        .get("url")
                        .and_then(Json::as_str)
                        .is_none_or(str::is_empty),
                    Some("base64") => {
                        source
                            .get("media_type")
                            .and_then(Json::as_str)
                            .is_none_or(str::is_empty)
                            || source
                                .get("data")
                                .and_then(Json::as_str)
                                .is_none_or(str::is_empty)
                    }
                    _ => true,
                }
            } else {
                object.values().any(contains_invalid_anthropic_image_source)
            }
        }
        Json::Array(items) => items.iter().any(contains_invalid_anthropic_image_source),
        _ => false,
    }
}

fn contains_any_key_recursive(value: &Json, keys: &[&str]) -> bool {
    match value {
        Json::Object(object) => {
            object.keys().any(|key| keys.contains(&key.as_str()))
                || object
                    .values()
                    .any(|value| contains_any_key_recursive(value, keys))
        }
        Json::Array(items) => items
            .iter()
            .any(|value| contains_any_key_recursive(value, keys)),
        _ => false,
    }
}

fn ensure_portable_response(protocol: WireProtocol, response: &Json) -> Result<()> {
    let unsupported =
        match protocol {
            WireProtocol::OpenaiChat => {
                response["choices"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|choice| {
                        choice["message"].get("audio").is_some()
                            || choice["message"].get("reasoning_content").is_some()
                    })
            }
            WireProtocol::OpenaiResponses => response["output"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|item| {
                    matches!(
                        item.get("type").and_then(Json::as_str),
                        Some(
                            "reasoning"
                                | "computer_call"
                                | "computer_call_output"
                                | "web_search_call"
                        )
                    )
                }),
            WireProtocol::AnthropicMessages => response["content"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|block| {
                    !matches!(
                        block.get("type").and_then(Json::as_str),
                        Some("text" | "tool_use")
                    )
                }),
        };
    if unsupported {
        Err(FlowError::InvalidArgument(
            "provider-specific response extension cannot be translated safely".into(),
        ))
    } else {
        Ok(())
    }
}
