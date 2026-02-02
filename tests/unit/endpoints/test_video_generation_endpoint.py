# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for VideoGenerationEndpoint."""

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.openai_video_generation import VideoGenerationEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestVideoGenerationEndpoint:
    """Tests for VideoGenerationEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for video generation."""
        return create_model_endpoint(
            EndpointType.VIDEO_GENERATION, model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        )

    @pytest.fixture
    def streaming_model_endpoint(self):
        """Create a test ModelEndpointInfo with streaming enabled."""
        return create_model_endpoint(
            EndpointType.VIDEO_GENERATION, model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", streaming=True
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a VideoGenerationEndpoint instance."""
        return create_endpoint_with_mock_transport(
            VideoGenerationEndpoint, model_endpoint
        )

    @pytest.fixture
    def streaming_endpoint(self, streaming_model_endpoint):
        """Create a VideoGenerationEndpoint instance with streaming."""
        return create_endpoint_with_mock_transport(
            VideoGenerationEndpoint, streaming_model_endpoint
        )

    # ===== format_payload tests =====

    def test_format_payload_simple_prompt(self, endpoint, model_endpoint):
        """Test simple prompt formatting."""
        turn = Turn(
            texts=[Text(contents=["A cat playing piano"])],
            model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["prompt"] == "A cat playing piano"
        assert payload["model"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        assert payload["response_format"] == "url"
        assert "stream" not in payload

    def test_format_payload_with_streaming(
        self, streaming_endpoint, streaming_model_endpoint
    ):
        """Test payload formatting with streaming enabled."""
        turn = Turn(
            texts=[Text(contents=["A dog running in a field"])],
            model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        )
        request_info = create_request_info(
            model_endpoint=streaming_model_endpoint, turns=[turn]
        )

        payload = streaming_endpoint.format_payload(request_info)

        assert payload["stream"] is True
        assert payload["prompt"] == "A dog running in a field"

    def test_format_payload_with_extra_inputs(self):
        """Test payload formatting with extra inputs for video parameters."""
        model_endpoint_with_extra = create_model_endpoint(
            EndpointType.VIDEO_GENERATION,
            model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            extra=[
                ("size", "1280x720"),
                ("num_frames", 32),
                ("fps", 24),
                ("num_inference_steps", 12),
                ("seed", 42),
            ],
        )
        endpoint = create_endpoint_with_mock_transport(
            VideoGenerationEndpoint, model_endpoint_with_extra
        )

        turn = Turn(
            texts=[Text(contents=["A bird flying"])],
            model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint_with_extra, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["size"] == "1280x720"
        assert payload["num_frames"] == 32
        assert payload["fps"] == 24
        assert payload["num_inference_steps"] == 12
        assert payload["seed"] == 42

    def test_format_payload_model_from_turn(self, endpoint, model_endpoint):
        """Test that turn model overrides endpoint model."""
        turn = Turn(
            texts=[Text(contents=["A tree swaying"])],
            model="custom-video-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "custom-video-model"

    def test_format_payload_no_turns_raises(self, endpoint, model_endpoint):
        """Test that missing turns raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_no_text_raises(self, endpoint, model_endpoint):
        """Test that missing text raises ValueError."""
        turn = Turn(texts=[], model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="requires text prompt"):
            endpoint.format_payload(request_info)

    def test_format_payload_empty_text_contents_raises(self, endpoint, model_endpoint):
        """Test that empty text contents raises ValueError."""
        turn = Turn(texts=[Text(contents=[])], model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="requires text prompt"):
            endpoint.format_payload(request_info)

    # ===== parse_response tests =====

    @pytest.mark.parametrize(
        "json_data,expected_videos",
        [
            # Simple SGLang response format (id at root)
            (
                {"id": "video_123", "url": "https://example.com/video.mp4"},
                [{"video_id": "video_123", "url": "https://example.com/video.mp4"}],
            ),
            # OpenAI-compatible response format with data array
            (
                {"data": [{"id": "video_456", "url": "https://example.com/video2.mp4"}]},
                [{"video_id": "video_456", "url": "https://example.com/video2.mp4"}],
            ),
            # Response with duration, frames, fps
            (
                {
                    "data": [
                        {
                            "id": "video_789",
                            "url": "https://example.com/video3.mp4",
                            "duration": 10.5,
                            "size": "1280x720",
                            "frames": 252,
                            "fps": 24,
                        }
                    ]
                },
                [
                    {
                        "video_id": "video_789",
                        "url": "https://example.com/video3.mp4",
                        "duration": 10.5,
                        "size": "1280x720",
                        "frames": 252,
                        "fps": 24,
                    }
                ],
            ),
            # Empty data array
            ({"data": []}, []),
        ],
    )  # fmt: skip
    def test_parse_response_video_formats(self, endpoint, json_data, expected_videos):
        """Test parsing various video format responses."""
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.videos) == len(expected_videos)
        for i, expected in enumerate(expected_videos):
            assert parsed.data.videos[i].video_id == expected.get("video_id")
            assert parsed.data.videos[i].url == expected.get("url")
            if "duration" in expected:
                assert parsed.data.videos[i].duration == expected["duration"]
            if "size" in expected:
                assert parsed.data.videos[i].size == expected["size"]
            if "frames" in expected:
                assert parsed.data.videos[i].frames == expected["frames"]
            if "fps" in expected:
                assert parsed.data.videos[i].fps == expected["fps"]

    def test_parse_response_with_revised_prompt(self, endpoint):
        """Test parsing response with revised prompt."""
        json_data = {
            "data": [
                {
                    "id": "video_123",
                    "url": "https://example.com/video.mp4",
                    "revised_prompt": "A beautiful cat playing piano, cinematic lighting",
                }
            ]
        }  # fmt: skip
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.videos) == 1
        assert (
            parsed.data.videos[0].revised_prompt
            == "A beautiful cat playing piano, cinematic lighting"
        )

    def test_parse_response_with_model_info(self, endpoint):
        """Test parsing response with model information."""
        json_data = {
            "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "data": [{"id": "video_123", "url": "https://example.com/video.mp4"}],
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.model == "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    def test_parse_response_with_usage(self, endpoint):
        """Test parsing response with usage information."""
        json_data = {
            "data": [{"id": "video_123", "url": "https://example.com/video.mp4"}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 0,
                "total_tokens": 50,
            },
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 50
        assert parsed.usage.total_tokens == 50

    def test_parse_response_perf_ns(self, endpoint):
        """Test that perf_ns is preserved."""
        json_data = {"id": "video_123", "url": "https://example.com/video.mp4"}
        mock_response = create_mock_response(perf_ns=888777666, json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 888777666

    def test_parse_response_extra_fields_ignored(self, endpoint):
        """Test that extra fields in response are ignored."""
        json_data = {
            "data": [
                {
                    "id": "video_123",
                    "url": "https://example.com/video.mp4",
                    "extra_field": "ignored",
                    "another_field": 123,
                }
            ],
            "model": "video-model",
            "unknown_field": "value",
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.videos[0].video_id == "video_123"

    def test_parse_response_no_json(self, endpoint):
        """Test parsing when get_json returns None."""
        mock_response = create_mock_response(json_data=None)
        mock_response.get_raw.return_value = ""

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None
