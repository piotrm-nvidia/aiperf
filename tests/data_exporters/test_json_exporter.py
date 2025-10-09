# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import EndpointType
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.json_exporter import JsonExporter


class TestJsonExporter:
    @pytest.fixture
    def sample_records(self):
        return [
            MetricResult(
                tag="ttft",
                header="Time to First Token",
                unit="ns",
                avg=123.0 * NANOS_PER_MILLIS,
                min=100.0 * NANOS_PER_MILLIS,
                max=150.0 * NANOS_PER_MILLIS,
                p1=101.0 * NANOS_PER_MILLIS,
                p5=105.0 * NANOS_PER_MILLIS,
                p10=108.0 * NANOS_PER_MILLIS,
                p25=110.0 * NANOS_PER_MILLIS,
                p50=120.0 * NANOS_PER_MILLIS,
                p75=130.0 * NANOS_PER_MILLIS,
                p90=140.0 * NANOS_PER_MILLIS,
                p95=None,
                p99=149.0 * NANOS_PER_MILLIS,
                std=10.0 * NANOS_PER_MILLIS,
                count=100,
            )
        ]

    @pytest.fixture
    def mock_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="custom_endpoint",
            )
        )

    @pytest.fixture
    def mock_results(self, sample_records):
        class MockResults:
            def __init__(self, metrics):
                self.metrics = metrics
                self.start_ns = None
                self.end_ns = None

            @property
            def records(self):
                return self.metrics

            @property
            def has_results(self):
                return bool(self.metrics)

            @property
            def was_cancelled(self):
                return False

            @property
            def error_summary(self):
                return []

        return MockResults(sample_records)

    @pytest.mark.asyncio
    async def test_json_exporter_creates_expected_json(
        self, mock_results, mock_user_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            assert expected_file.exists()

            with open(expected_file) as f:
                data = json.load(f)

            # Verify GenAI-Perf compatible structure: metrics at top level
            assert "ttft" in data
            ttft_metric = data["ttft"]
            assert isinstance(ttft_metric, dict)
            
            # Verify metric has expected fields and values (converted to display units)
            assert ttft_metric["unit"] == "ms"
            assert ttft_metric["avg"] == 123.0
            assert ttft_metric["min"] == 100.0
            assert ttft_metric["max"] == 150.0
            assert ttft_metric["p1"] == 101.0
            assert ttft_metric["p5"] == 105.0
            assert ttft_metric["p10"] == 108.0  # New p10 percentile
            assert ttft_metric["p25"] == 110.0
            assert ttft_metric["p50"] == 120.0
            assert ttft_metric["p75"] == 130.0
            assert ttft_metric["p90"] == 140.0
            assert ttft_metric["p99"] == 149.0
            assert ttft_metric["std"] == 10.0
            
            # Verify metric does NOT have tag, header, or count (GenAI-Perf format)
            assert "tag" not in ttft_metric
            assert "header" not in ttft_metric
            assert "count" not in ttft_metric

            # Verify input_config is present
            assert "input_config" in data
            assert isinstance(data["input_config"], dict)

            # Verify execution_metadata exists and contains expected fields
            assert "execution_metadata" in data
            execution_metadata = data["execution_metadata"]
            assert isinstance(execution_metadata, dict)
            assert "was_cancelled" in execution_metadata
            assert execution_metadata["was_cancelled"] is False
            assert "error_summary" in execution_metadata

            # Verify ai_perf_v1 metadata contains tag, header, and count
            assert "ai_perf_v1" in data
            ai_perf_v1 = data["ai_perf_v1"]
            assert isinstance(ai_perf_v1, dict)
            assert "ttft" in ai_perf_v1
            
            ttft_metadata = ai_perf_v1["ttft"]
            assert ttft_metadata["tag"] == "ttft"
            assert ttft_metadata["header"] == "Time to First Token"
            assert ttft_metadata["count"] == 100

            # Verify old "records" structure is NOT present
            assert "records" not in data
