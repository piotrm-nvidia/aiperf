# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import aiofiles
from pydantic import BaseModel

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

    def _inject_genai_perf_fields(self, export_data: dict):
        tags_map_for_genai_perf = {
            "ttft": "time_to_first_token",
            "ttst": "time_to_second_token",
        }

        allowed_genai_perf_tags_full_stat_tags = {
            "request_latency",
            "time_to_first_token",
            "time_to_second_token",
            "inter_token_latency",
            "output_token_throughput_per_user",
            "output_sequence_length",
            "input_sequence_length",
        }

        allowed_genai_perf_tags_avg_tags = {
            "request_throughput",
            "request_count",
            "output_token_throughput",
        }

        # Add individual metrics at top level (GenAI-Perf format)
        for tag, metric_result in export_data["records"].items():
            if tag in tags_map_for_genai_perf:
                mapped_tag = tags_map_for_genai_perf[tag]
            else:
                mapped_tag = tag
            genai_metric = None
            if mapped_tag in allowed_genai_perf_tags_full_stat_tags:
                genai_metric = {}
                for metric in metric_result:
                    if metric in {"std", "unit", "avg", "min", "max"} or metric.startswith("p"):
                        genai_metric[metric] = metric_result[metric]
            elif mapped_tag in allowed_genai_perf_tags_avg_tags:
                genai_metric = {}
                for metric in metric_result:
                    if metric in {"avg", "unit"}:
                        genai_metric[metric] = metric_result[metric]

            if genai_metric:
                export_data[mapped_tag] = genai_metric




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

        export_data = JsonExportData(
            input_config=self._input_config,
            records=converted_records,
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
        )

        export_data_json = export_data.model_dump_json(indent=2, exclude_unset=True)

        import json
        import os
        if os.getenv("AIPERF_GENAI_PERF_INJECT", "false") == "true":
            export_data_dict = json.loads(export_data_json)
            self._inject_genai_perf_fields(export_data_dict)
            export_data_json = json.dumps(export_data_dict, indent=2)


        self.debug(lambda: f"Exporting data to JSON file: {export_data}")
        
        async with aiofiles.open(self._file_path, "w") as f:
            await f.write(export_data_json)
