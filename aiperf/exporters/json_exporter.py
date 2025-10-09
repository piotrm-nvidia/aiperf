# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any

import aiofiles
from pydantic import BaseModel, Field

from aiperf.common.config import UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType, MetricFlags
from aiperf.common.factories import DataExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ErrorDetailsCount, MetricResult
from aiperf.common.protocols import DataExporterProtocol
from aiperf.common.types import MetricTagT
from aiperf.exporters.display_units_utils import convert_all_metrics_to_display_units
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.metrics.metric_registry import MetricRegistry


class GenAIPerfMetric(BaseModel):
    """Metric in GenAI-Perf format (without tag, header, count)."""

    unit: str
    avg: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    std: float | None = None


class AIPerfV1Metadata(BaseModel):
    """AIPerf-specific metadata for a metric."""

    tag: MetricTagT
    header: str
    count: int | None = None


class ExecutionMetadata(BaseModel):
    """Execution-related metadata."""

    was_cancelled: bool | None = None
    error_summary: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


class JsonExportData(BaseModel):
    """Data to be exported to a JSON file."""

    records: dict[MetricTagT, MetricResult] | None = None
    input_config: UserConfig | None = None
    was_cancelled: bool | None = None
    error_summary: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


@DataExporterFactory.register(DataExporterType.JSON)
@implements_protocol(DataExporterProtocol)
class JsonExporter(AIPerfLoggerMixin):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.debug(lambda: f"Initializing JsonExporter with config: {exporter_config}")
        self._results = exporter_config.results
        self._output_directory = exporter_config.user_config.output.artifact_directory
        self._input_config = exporter_config.user_config
        self._metric_registry = MetricRegistry
        self._file_path = (
            self._output_directory / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="JSON Export",
            file_path=self._file_path,
        )

    def _should_export(self, metric: MetricResult) -> bool:
        """Check if a metric should be exported."""
        metric_class = MetricRegistry.get_class(metric.tag)
        res = metric_class.missing_flags(
            MetricFlags.EXPERIMENTAL | MetricFlags.INTERNAL
        )
        self.debug(lambda: f"Metric '{metric.tag}' should be exported: {res}")
        return res

    async def export(self) -> None:
        self._output_directory.mkdir(parents=True, exist_ok=True)

        start_time = (
            datetime.fromtimestamp(self._results.start_ns / NANOS_PER_SECOND)
            if self._results.start_ns
            else None
        )
        end_time = (
            datetime.fromtimestamp(self._results.end_ns / NANOS_PER_SECOND)
            if self._results.end_ns
            else None
        )

        converted_records: dict[MetricTagT, MetricResult] = {}
        if self._results.records:
            converted_records = convert_all_metrics_to_display_units(
                self._results.records, self._metric_registry
            )
            converted_records = {
                k: v for k, v in converted_records.items() if self._should_export(v)
            }

        # Build GenAI-Perf compatible structure
        export_dict: dict[str, Any] = {}

        # Add individual metrics at top level (GenAI-Perf format)
        for tag, metric_result in converted_records.items():
            genai_metric = GenAIPerfMetric(
                unit=metric_result.unit,
                avg=metric_result.avg,
                min=metric_result.min,
                max=metric_result.max,
                p1=metric_result.p1,
                p5=metric_result.p5,
                p10=metric_result.p10,
                p25=metric_result.p25,
                p50=metric_result.p50,
                p75=metric_result.p75,
                p90=metric_result.p90,
                p95=metric_result.p95,
                p99=metric_result.p99,
                std=metric_result.std,
            )
            export_dict[tag] = genai_metric.model_dump(exclude_unset=True)

        # Add input_config
        if self._input_config:
            export_dict["input_config"] = self._input_config.model_dump(
                exclude_unset=True
            )

        # Add execution_metadata
        execution_metadata = ExecutionMetadata(
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
        )
        export_dict["execution_metadata"] = execution_metadata.model_dump(
            exclude_unset=True
        )

        # Add ai_perf_v1 metadata
        ai_perf_v1_metadata: dict[str, Any] = {}
        for tag, metric_result in converted_records.items():
            metadata = AIPerfV1Metadata(
                tag=metric_result.tag,
                header=metric_result.header,
                count=metric_result.count,
            )
            ai_perf_v1_metadata[tag] = metadata.model_dump(exclude_unset=True)
        export_dict["ai_perf_v1"] = ai_perf_v1_metadata

        self.debug(lambda: f"Exporting data to JSON file")
        # Use orjson for better control and to maintain order
        import orjson

        export_data_json = orjson.dumps(
            export_dict, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        ).decode("utf-8")

        async with aiofiles.open(self._file_path, "w") as f:
            await f.write(export_data_json)
