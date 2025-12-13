"""Progress tracking for document processing."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStage:
    """Represents a processing stage."""

    name: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate stage duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ProgressTracker:
    """
    Track document processing progress.

    Features:
    - Stage-by-stage progress tracking
    - Page-by-page progress for large documents
    - Time estimation
    - Error/warning logging
    """

    def __init__(self, document_name: str, total_pages: Optional[int] = None):
        """
        Initialize progress tracker.

        Args:
            document_name: Name of document being processed
            total_pages: Total number of pages (if applicable)
        """
        self.document_name = document_name
        self.total_pages = total_pages
        self.current_page = 0

        self.stages: Dict[str, ProcessingStage] = {}
        self.current_stage: Optional[str] = None

        self.started_at = datetime.now()
        self.completed_at: Optional[datetime] = None

        self.errors: list[str] = []
        self.warnings: list[str] = []

        logger.info(
            f"ProgressTracker initialized for: {document_name} "
            f"({total_pages or 'unknown'} pages)"
        )

    def start_stage(self, stage_name: str, details: Optional[Dict[str, Any]] = None):
        """
        Start a processing stage.

        Args:
            stage_name: Name of the stage
            details: Optional stage details
        """
        stage = ProcessingStage(
            name=stage_name,
            started_at=datetime.now(),
            status="in_progress",
            details=details or {},
        )

        self.stages[stage_name] = stage
        self.current_stage = stage_name

        logger.info(f"Stage started: {stage_name}")

    def complete_stage(
        self, stage_name: str, details: Optional[Dict[str, Any]] = None
    ):
        """
        Complete a processing stage.

        Args:
            stage_name: Name of the stage
            details: Optional completion details
        """
        if stage_name not in self.stages:
            logger.warning(f"Stage not found: {stage_name}")
            return

        stage = self.stages[stage_name]
        stage.completed_at = datetime.now()
        stage.status = "completed"

        if details:
            stage.details.update(details)

        duration = stage.duration_seconds
        logger.info(f"Stage completed: {stage_name} ({duration:.2f}s)")

        # Move to next stage
        if self.current_stage == stage_name:
            self.current_stage = None

    def fail_stage(self, stage_name: str, error_message: str):
        """
        Mark a stage as failed.

        Args:
            stage_name: Name of the stage
            error_message: Error description
        """
        if stage_name not in self.stages:
            logger.warning(f"Stage not found: {stage_name}")
            return

        stage = self.stages[stage_name]
        stage.completed_at = datetime.now()
        stage.status = "failed"
        stage.details["error"] = error_message

        self.errors.append(f"[{stage_name}] {error_message}")
        logger.error(f"Stage failed: {stage_name} - {error_message}")

    def update_page_progress(self, current_page: int):
        """
        Update page processing progress.

        Args:
            current_page: Current page being processed
        """
        self.current_page = current_page

        if self.total_pages:
            progress_pct = (current_page / self.total_pages) * 100
            logger.info(
                f"Page progress: {current_page}/{self.total_pages} ({progress_pct:.1f}%)"
            )
        else:
            logger.info(f"Page progress: {current_page}")

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)

    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        logger.error(message)

    def complete(self):
        """Mark overall processing as complete."""
        self.completed_at = datetime.now()

        # Complete any in-progress stages
        if self.current_stage and self.current_stage in self.stages:
            self.complete_stage(self.current_stage)

        duration = self.total_duration_seconds
        logger.info(f"Processing complete: {self.document_name} ({duration:.2f}s)")

    @property
    def total_duration_seconds(self) -> float:
        """Calculate total processing duration."""
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimate remaining time based on current progress."""
        if not self.total_pages or self.current_page == 0:
            return None

        elapsed = self.total_duration_seconds
        pages_processed = self.current_page
        pages_remaining = self.total_pages - pages_processed

        if pages_remaining <= 0:
            return 0

        avg_time_per_page = elapsed / pages_processed
        return avg_time_per_page * pages_remaining

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status summary.

        Returns:
            Status dictionary with progress info
        """
        eta = self.estimated_time_remaining
        eta_str = f"{eta:.1f}s" if eta is not None else "unknown"

        return {
            "document": self.document_name,
            "current_stage": self.current_stage,
            "pages_processed": self.current_page,
            "total_pages": self.total_pages,
            "progress_pct": (
                round((self.current_page / self.total_pages) * 100, 1)
                if self.total_pages
                else None
            ),
            "elapsed_seconds": round(self.total_duration_seconds, 2),
            "eta_seconds": round(eta, 2) if eta is not None else None,
            "eta_readable": eta_str,
            "stages_completed": sum(
                1 for s in self.stages.values() if s.status == "completed"
            ),
            "total_stages": len(self.stages),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "is_complete": self.completed_at is not None,
        }

    def print_status(self):
        """Print current status to console."""
        status = self.get_status()

        print("\n" + "=" * 70)
        print(f"PROCESSING: {status['document']}")
        print("=" * 70)

        if status["total_pages"]:
            print(
                f"Pages: {status['pages_processed']}/{status['total_pages']} "
                f"({status['progress_pct']}%)"
            )
        else:
            print(f"Pages: {status['pages_processed']}")

        print(
            f"Stages: {status['stages_completed']}/{status['total_stages']} completed"
        )

        if status["current_stage"]:
            print(f"Current Stage: {status['current_stage']}")

        print(f"Elapsed: {status['elapsed_seconds']}s")

        if status["eta_seconds"] is not None:
            print(f"ETA: {status['eta_readable']}")

        if status["errors"]:
            print(f"\n⚠ Errors: {status['errors']}")
        if status["warnings"]:
            print(f"⚠ Warnings: {status['warnings']}")

        print("=" * 70 + "\n")

    def print_summary(self):
        """Print final processing summary."""
        print("\n" + "=" * 70)
        print(f"PROCESSING SUMMARY: {self.document_name}")
        print("=" * 70)

        total_duration = self.total_duration_seconds
        print(f"Total Duration: {total_duration:.2f}s")

        if self.total_pages:
            avg_per_page = total_duration / self.total_pages
            print(f"Average per Page: {avg_per_page:.2f}s")

        print(f"\nStages ({len(self.stages)}):")
        for stage_name, stage in self.stages.items():
            status_icon = {
                "completed": "✓",
                "failed": "✗",
                "in_progress": "→",
                "pending": "○",
            }.get(stage.status, "?")

            duration_str = (
                f"{stage.duration_seconds:.2f}s"
                if stage.duration_seconds
                else "N/A"
            )

            print(f"  [{status_icon}] {stage_name}: {stage.status} ({duration_str})")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        print("=" * 70 + "\n")
