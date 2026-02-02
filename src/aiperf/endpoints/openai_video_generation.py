# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
    VideoDataItem,
    VideoResponseData,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class VideoGenerationEndpoint(BaseEndpoint):
    """OpenAI Video Generation endpoint.

    Supports video generation from text prompts using models like Sora or SGLang video models.
    Handles both streaming and non-streaming responses.

    See: https://platform.openai.com/docs/api-reference/videos
    See: https://docs.sglang.io/basic_usage/openai_api.html
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Video Generation request payload from RequestInfo.

        Supports all OpenAI Video Generation API parameters:
        - prompt (required): Text description from turn.texts[0]
        - model (optional): From turn.model or model_endpoint.primary_model_name
        - stream (optional): From model_endpoint.endpoint.streaming
        - size, num_frames, fps, num_inference_steps, seed, response_format:
          Pass via --extra-inputs "input_name:value"

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Video Generation API payload with all specified parameters
        """
        if not request_info.turns:
            raise ValueError("Video generation endpoint requires at least one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        if not turn.texts or not turn.texts[0].contents:
            raise ValueError(
                "Video generation endpoint requires text prompt in first turn."
            )

        prompt = turn.texts[0].contents[0]

        # NOTE: response_format is set to url by default for SGLang video generation
        payload = {
            "prompt": prompt,
            "model": turn.model or model_endpoint.primary_model_name,
            "response_format": "url",
        }

        if model_endpoint.endpoint.streaming:
            payload["stream"] = True

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Video Generation response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted video data and usage info
        """
        json_obj = response.get_json()
        if not json_obj:
            self.debug(
                lambda: f"No JSON object found in response: {response.get_raw()}"
            )
            return None

        videos = []

        # Handle different response formats
        if "id" in json_obj and "data" not in json_obj:
            # SGLang video generation response format (simple format with id at root)
            videos.append(
                VideoDataItem(
                    video_id=json_obj.get("id"),
                    url=json_obj.get("url"),
                )
            )
        elif "data" in json_obj:
            # OpenAI-compatible response format with data array
            for item in json_obj.get("data", []):
                videos.append(
                    VideoDataItem(
                        video_id=item.get("id"),
                        url=item.get("url"),
                        duration=item.get("duration"),
                        size=item.get("size"),
                        format=item.get("format"),
                        revised_prompt=item.get("revised_prompt"),
                        frames=item.get("frames"),
                        fps=item.get("fps"),
                    )
                )

        response_data = VideoResponseData(
            videos=videos,
            model=json_obj.get("model"),
        )

        usage = json_obj.get("usage") or None

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=response_data,
            usage=usage
        )
