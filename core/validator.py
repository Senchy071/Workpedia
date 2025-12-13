"""Validation layer for document processing results."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    severity: str  # error, warning, info
    category: str  # structure, table, metadata, content
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""

    is_valid: bool
    total_checks: int
    passed_checks: int
    issues: List[ValidationIssue]
    summary: Dict[str, Any]


class DocumentValidator:
    """
    Validates document processing results.

    Features:
    - Structure extraction completeness
    - Table integrity verification
    - Metadata completeness checks
    - Parsing error detection
    - Validation reporting
    """

    def __init__(
        self,
        require_text: bool = True,
        min_text_length: int = 10,
        require_metadata: bool = True,
    ):
        """
        Initialize document validator.

        Args:
            require_text: Require non-empty text content
            min_text_length: Minimum text length to consider valid
            require_metadata: Require metadata to be present
        """
        self.require_text = require_text
        self.min_text_length = min_text_length
        self.require_metadata = require_metadata

        logger.info("DocumentValidator initialized")

    def validate(self, parse_result: Dict[str, Any]) -> ValidationReport:
        """
        Validate document parsing result.

        Args:
            parse_result: Result from document parser

        Returns:
            ValidationReport with findings
        """
        logger.info(f"Validating document: {parse_result.get('doc_id')}")

        issues = []
        checks_performed = 0
        checks_passed = 0

        # Validate metadata
        metadata_issues, meta_checks, meta_passed = self._validate_metadata(
            parse_result
        )
        issues.extend(metadata_issues)
        checks_performed += meta_checks
        checks_passed += meta_passed

        # Validate text content
        text_issues, text_checks, text_passed = self._validate_text_content(
            parse_result
        )
        issues.extend(text_issues)
        checks_performed += text_checks
        checks_passed += text_passed

        # Validate structure
        struct_issues, struct_checks, struct_passed = self._validate_structure(
            parse_result
        )
        issues.extend(struct_issues)
        checks_performed += struct_checks
        checks_passed += struct_passed

        # Validate tables (if present)
        table_issues, table_checks, table_passed = self._validate_tables(parse_result)
        issues.extend(table_issues)
        checks_performed += table_checks
        checks_passed += table_passed

        # Determine if valid (no error-level issues)
        error_count = sum(1 for issue in issues if issue.severity == "error")
        is_valid = error_count == 0

        # Build summary
        summary = {
            "errors": error_count,
            "warnings": sum(1 for issue in issues if issue.severity == "warning"),
            "info": sum(1 for issue in issues if issue.severity == "info"),
            "pass_rate": (
                round((checks_passed / checks_performed) * 100, 2)
                if checks_performed > 0
                else 0
            ),
        }

        report = ValidationReport(
            is_valid=is_valid,
            total_checks=checks_performed,
            passed_checks=checks_passed,
            issues=issues,
            summary=summary,
        )

        logger.info(
            f"Validation complete: {checks_passed}/{checks_performed} checks passed, "
            f"{error_count} errors, {summary['warnings']} warnings"
        )

        return report

    def _validate_metadata(
        self, parse_result: Dict[str, Any]
    ) -> tuple[List[ValidationIssue], int, int]:
        """Validate metadata completeness."""
        issues = []
        checks = 0
        passed = 0

        # Check metadata exists
        checks += 1
        if "metadata" not in parse_result:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="metadata",
                    message="Missing metadata field",
                )
            )
        else:
            passed += 1
            metadata = parse_result["metadata"]

            # Check required fields
            required_fields = [
                "filename",
                "file_size_mb",
                "processing_time_seconds",
                "parsed_at",
            ]

            for field in required_fields:
                checks += 1
                if field not in metadata:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            category="metadata",
                            message=f"Missing metadata field: {field}",
                        )
                    )
                else:
                    passed += 1

            # Check for reasonable values
            checks += 1
            if metadata.get("file_size_mb", 0) <= 0:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="metadata",
                        message="Invalid file size in metadata",
                        details={"file_size_mb": metadata.get("file_size_mb")},
                    )
                )
            else:
                passed += 1

        return issues, checks, passed

    def _validate_text_content(
        self, parse_result: Dict[str, Any]
    ) -> tuple[List[ValidationIssue], int, int]:
        """Validate text content extraction."""
        issues = []
        checks = 0
        passed = 0

        if self.require_text:
            checks += 1
            raw_text = parse_result.get("raw_text", "")

            if not raw_text:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="content",
                        message="No text content extracted",
                    )
                )
            else:
                passed += 1

                # Check minimum length
                checks += 1
                if len(raw_text) < self.min_text_length:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            category="content",
                            message=f"Text content too short (< {self.min_text_length} chars)",
                            details={"text_length": len(raw_text)},
                        )
                    )
                else:
                    passed += 1

        return issues, checks, passed

    def _validate_structure(
        self, parse_result: Dict[str, Any]
    ) -> tuple[List[ValidationIssue], int, int]:
        """Validate structure extraction."""
        issues = []
        checks = 0
        passed = 0

        # Check structure field exists
        checks += 1
        if "structure" not in parse_result:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="structure",
                    message="Missing structure field",
                )
            )
        else:
            passed += 1
            structure = parse_result["structure"]

            # Check pages count
            checks += 1
            pages = structure.get("pages", 0)
            if pages == 0:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        category="structure",
                        message="Document has 0 pages (may be non-paginated format)",
                    )
                )
            else:
                passed += 1

        return issues, checks, passed

    def _validate_tables(
        self, parse_result: Dict[str, Any]
    ) -> tuple[List[ValidationIssue], int, int]:
        """Validate table extraction and integrity."""
        issues = []
        checks = 0
        passed = 0

        # Check if structure analysis was performed
        if "structure_analysis" in parse_result:
            structure_analysis = parse_result["structure_analysis"]

            if "tables" in structure_analysis:
                tables = structure_analysis["tables"]

                # Validate each table
                for idx, table in enumerate(tables):
                    checks += 1

                    # Check table has ID
                    if "table_id" not in table:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                category="table",
                                message=f"Table {idx} missing ID",
                            )
                        )
                    else:
                        passed += 1

                    # Check dimensions
                    checks += 1
                    rows = table.get("rows", 0)
                    cols = table.get("cols", 0)

                    if rows == 0 or cols == 0:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                category="table",
                                message=f"Table {table.get('table_id', idx)} has "
                                f"invalid dimensions: {rows}x{cols}",
                                details={"rows": rows, "cols": cols},
                            )
                        )
                    else:
                        passed += 1

        return issues, checks, passed

    def print_report(self, report: ValidationReport):
        """Print validation report to console."""
        print("\n" + "=" * 70)
        print("DOCUMENT VALIDATION REPORT")
        print("=" * 70)
        print(f"Overall Status: {'✓ VALID' if report.is_valid else '✗ INVALID'}")
        print(f"Checks Passed: {report.passed_checks}/{report.total_checks}")
        print(f"Pass Rate: {report.summary['pass_rate']}%")
        print("\n" + "-" * 70)
        print(
            f"Errors: {report.summary['errors']} | "
            f"Warnings: {report.summary['warnings']} | "
            f"Info: {report.summary['info']}"
        )

        if report.issues:
            print("\n" + "-" * 70)
            print("ISSUES:")
            for idx, issue in enumerate(report.issues, 1):
                severity_icon = {
                    "error": "✗",
                    "warning": "⚠",
                    "info": "ℹ",
                }.get(issue.severity, "•")

                print(
                    f"{idx}. [{severity_icon} {issue.severity.upper()}] "
                    f"[{issue.category}] {issue.message}"
                )
                if issue.details:
                    print(f"   Details: {issue.details}")

        print("=" * 70 + "\n")
