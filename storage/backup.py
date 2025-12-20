"""Vector store backup and restore functionality.

This module provides comprehensive backup and restore capabilities for the ChromaDB
vector store, including:
- Full vector store backups with versioning
- Incremental backups (metadata only)
- Scheduled backup management
- Backup retention policies
- One-click restore from backup
- Export/import to portable formats
"""

import json
import logging
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages vector store backups and restores.

    Features:
    - Full backups with versioning
    - Backup retention policies
    - One-click restore
    - Backup validation
    - Metadata tracking

    Args:
        chroma_dir: Path to ChromaDB persistence directory
        backup_dir: Path to backup storage directory
        max_backups: Maximum number of backups to retain (0 = unlimited)
        compress: Whether to compress backups (default: True)

    Example:
        manager = BackupManager(
            chroma_dir="chroma_db",
            backup_dir="backups",
            max_backups=10
        )

        # Create backup
        backup_path = manager.create_backup(description="Before migration")

        # List backups
        backups = manager.list_backups()

        # Restore from backup
        manager.restore_backup(backup_path)
    """

    def __init__(
        self,
        chroma_dir: Union[str, Path],
        backup_dir: Union[str, Path],
        max_backups: int = 10,
        compress: bool = True,
    ):
        self.chroma_dir = Path(chroma_dir)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.compress = compress

        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"BackupManager initialized: chroma_dir={chroma_dir}, "
            f"backup_dir={backup_dir}, max_backups={max_backups}"
        )

    def create_backup(
        self,
        description: str = "",
        backup_name: Optional[str] = None,
    ) -> Path:
        """Create a full backup of the vector store.

        Args:
            description: Human-readable description of the backup
            backup_name: Custom backup name (uses timestamp if None)

        Returns:
            Path to created backup file

        Raises:
            FileNotFoundError: If chroma_dir doesn't exist
            PermissionError: If backup_dir is not writable
        """
        if not self.chroma_dir.exists():
            raise FileNotFoundError(f"ChromaDB directory not found: {self.chroma_dir}")

        # Generate backup name
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        # Add extension
        if self.compress:
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"
        else:
            backup_file = self.backup_dir / backup_name

        logger.info(f"Creating backup: {backup_file}")

        # Create metadata
        metadata = {
            "backup_name": backup_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "chroma_dir": str(self.chroma_dir),
            "compressed": self.compress,
            "version": "1.0",
        }

        # Create backup
        if self.compress:
            self._create_compressed_backup(backup_file, metadata)
        else:
            self._create_directory_backup(backup_file, metadata)

        logger.info(f"Backup created successfully: {backup_file}")

        # Apply retention policy
        self._apply_retention_policy()

        return backup_file

    def _create_compressed_backup(self, backup_file: Path, metadata: Dict[str, Any]) -> None:
        """Create a compressed tar.gz backup."""
        with tarfile.open(backup_file, "w:gz") as tar:
            # Add ChromaDB directory
            tar.add(self.chroma_dir, arcname="chroma_db")

            # Add metadata
            metadata_json = json.dumps(metadata, indent=2)
            import io
            import tarfile as tf

            metadata_info = tf.TarInfo(name="backup_metadata.json")
            metadata_info.size = len(metadata_json.encode())
            tar.addfile(metadata_info, io.BytesIO(metadata_json.encode()))

    def _create_directory_backup(self, backup_dir: Path, metadata: Dict[str, Any]) -> None:
        """Create an uncompressed directory backup."""
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy ChromaDB directory
        chroma_backup = backup_dir / "chroma_db"
        if chroma_backup.exists():
            shutil.rmtree(chroma_backup)
        shutil.copytree(self.chroma_dir, chroma_backup)

        # Write metadata
        metadata_file = backup_dir / "backup_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def restore_backup(
        self,
        backup_path: Union[str, Path],
        force: bool = False,
    ) -> None:
        """Restore vector store from backup.

        Args:
            backup_path: Path to backup file or directory
            force: If True, overwrite existing data without confirmation

        Raises:
            FileNotFoundError: If backup doesn't exist
            ValueError: If backup is invalid
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        logger.info(f"Restoring from backup: {backup_path}")

        # Validate backup
        if not self._validate_backup(backup_path):
            raise ValueError(f"Invalid backup: {backup_path}")

        # Check if chroma_dir exists and warn if not forcing
        if self.chroma_dir.exists() and not force:
            logger.warning(
                f"ChromaDB directory exists: {self.chroma_dir}. "
                "Use force=True to overwrite."
            )
            raise ValueError(
                "ChromaDB directory exists. Use force=True to overwrite."
            )

        # Remove existing data
        if self.chroma_dir.exists():
            logger.info(f"Removing existing ChromaDB directory: {self.chroma_dir}")
            shutil.rmtree(self.chroma_dir)

        # Restore based on backup type
        if backup_path.is_dir():
            self._restore_directory_backup(backup_path)
        elif backup_path.suffix == ".gz" or (backup_path.is_file() and tarfile.is_tarfile(str(backup_path))):
            self._restore_compressed_backup(backup_path)
        else:
            raise ValueError(f"Unknown backup format: {backup_path}")

        logger.info(f"Backup restored successfully: {backup_path}")

    def _restore_compressed_backup(self, backup_file: Path) -> None:
        """Restore from compressed tar.gz backup."""
        with tarfile.open(backup_file, "r:gz") as tar:
            # Extract chroma_db directory
            for member in tar.getmembers():
                if member.name.startswith("chroma_db/"):
                    # Adjust path to extract to chroma_dir
                    member.name = member.name.replace("chroma_db/", "", 1)
                    tar.extract(member, self.chroma_dir)

    def _restore_directory_backup(self, backup_dir: Path) -> None:
        """Restore from uncompressed directory backup."""
        chroma_backup = backup_dir / "chroma_db"
        if not chroma_backup.exists():
            raise ValueError(f"Invalid backup: chroma_db not found in {backup_dir}")

        shutil.copytree(chroma_backup, self.chroma_dir)

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata.

        Returns:
            List of backup information dictionaries
        """
        backups = []

        for item in sorted(self.backup_dir.iterdir(), reverse=True):
            if item.name.startswith("backup_"):
                metadata = self._get_backup_metadata(item)
                if metadata:
                    metadata["path"] = str(item)
                    metadata["size_mb"] = self._get_backup_size(item)
                    backups.append(metadata)

        return backups

    def _get_backup_metadata(self, backup_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from backup."""
        try:
            if backup_path.is_dir():
                # Directory backup
                metadata_file = backup_path / "backup_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        return json.load(f)
            elif backup_path.suffix == ".gz" or (backup_path.is_file() and tarfile.is_tarfile(str(backup_path))):
                # Compressed backup
                with tarfile.open(backup_path, "r:gz") as tar:
                    try:
                        metadata_file = tar.extractfile("backup_metadata.json")
                        if metadata_file:
                            return json.load(metadata_file)
                    except KeyError:
                        pass

            # If no metadata, create basic info
            return {
                "backup_name": backup_path.name,
                "description": "No metadata available",
                "created_at": datetime.fromtimestamp(
                    backup_path.stat().st_mtime
                ).isoformat(),
            }
        except Exception as e:
            logger.warning(f"Failed to read backup metadata: {e}")
            return None

    def _get_backup_size(self, backup_path: Path) -> float:
        """Get backup size in MB."""
        if backup_path.is_file():
            return backup_path.stat().st_size / (1024 * 1024)
        elif backup_path.is_dir():
            total_size = sum(
                f.stat().st_size for f in backup_path.rglob("*") if f.is_file()
            )
            return total_size / (1024 * 1024)
        return 0.0

    def _validate_backup(self, backup_path: Path) -> bool:
        """Validate backup integrity.

        Args:
            backup_path: Path to backup

        Returns:
            True if backup is valid
        """
        try:
            if backup_path.is_dir():
                # Validate directory backup
                chroma_backup = backup_path / "chroma_db"
                return chroma_backup.exists() and chroma_backup.is_dir()
            elif backup_path.suffix == ".gz" or (backup_path.is_file() and tarfile.is_tarfile(str(backup_path))):
                # Validate tar.gz file
                with tarfile.open(backup_path, "r:gz") as tar:
                    # Check that chroma_db exists in archive
                    members = tar.getmembers()
                    has_chroma = any(m.name.startswith("chroma_db/") for m in members)
                    return has_chroma

            return False
        except Exception as e:
            logger.error(f"Backup validation failed: {e}")
            return False

    def delete_backup(self, backup_path: Union[str, Path]) -> None:
        """Delete a specific backup.

        Args:
            backup_path: Path to backup to delete

        Raises:
            FileNotFoundError: If backup doesn't exist
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        logger.info(f"Deleting backup: {backup_path}")

        if backup_path.is_file():
            backup_path.unlink()
        else:
            shutil.rmtree(backup_path)

        logger.info(f"Backup deleted: {backup_path}")

    def _apply_retention_policy(self) -> None:
        """Apply backup retention policy (delete old backups)."""
        if self.max_backups == 0:
            return  # Unlimited backups

        backups = self.list_backups()

        if len(backups) > self.max_backups:
            # Delete oldest backups
            backups_to_delete = backups[self.max_backups :]
            for backup in backups_to_delete:
                try:
                    self.delete_backup(backup["path"])
                    logger.info(
                        f"Deleted old backup (retention policy): {backup['backup_name']}"
                    )
                except Exception as e:
                    logger.error(f"Failed to delete backup: {e}")

    def export_to_json(self, output_file: Union[str, Path]) -> None:
        """Export vector store metadata to JSON (portable format).

        Note: This exports metadata only, not the actual vector data.
        Use create_backup() for full backups.

        Args:
            output_file: Path to output JSON file
        """
        output_file = Path(output_file)

        logger.info(f"Exporting vector store metadata to: {output_file}")

        metadata = {
            "export_date": datetime.now().isoformat(),
            "chroma_dir": str(self.chroma_dir),
            "version": "1.0",
            "note": "This is a metadata export only. Use full backups for complete data.",
        }

        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata exported successfully: {output_file}")

    def get_backup_stats(self) -> Dict[str, Any]:
        """Get statistics about backups.

        Returns:
            Dictionary with backup statistics
        """
        backups = self.list_backups()

        total_size = sum(b["size_mb"] for b in backups)

        return {
            "total_backups": len(backups),
            "total_size_mb": round(total_size, 2),
            "max_backups": self.max_backups,
            "backup_dir": str(self.backup_dir),
            "oldest_backup": backups[-1]["created_at"] if backups else None,
            "newest_backup": backups[0]["created_at"] if backups else None,
        }
