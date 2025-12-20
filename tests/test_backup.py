"""Tests for vector store backup and restore functionality."""

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest

from storage.backup import BackupManager


class TestBackupManager:
    """Tests for BackupManager class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as chroma_dir, tempfile.TemporaryDirectory() as backup_dir:
            chroma_path = Path(chroma_dir)
            backup_path = Path(backup_dir)

            # Create sample ChromaDB structure
            (chroma_path / "chroma.sqlite3").write_text("fake db")
            (chroma_path / "data").mkdir()
            (chroma_path / "data" / "vectors.bin").write_text("fake vectors")

            yield chroma_path, backup_path

    @pytest.fixture
    def manager(self, temp_dirs):
        """Create BackupManager instance."""
        chroma_dir, backup_dir = temp_dirs
        return BackupManager(
            chroma_dir=chroma_dir,
            backup_dir=backup_dir,
            max_backups=5,
            compress=True,
        )

    def test_init(self, temp_dirs):
        """Test BackupManager initialization."""
        chroma_dir, backup_dir = temp_dirs
        manager = BackupManager(
            chroma_dir=chroma_dir,
            backup_dir=backup_dir,
            max_backups=10,
            compress=True,
        )

        assert manager.chroma_dir == chroma_dir
        assert manager.backup_dir == backup_dir
        assert manager.max_backups == 10
        assert manager.compress is True
        assert backup_dir.exists()

    def test_create_backup_compressed(self, manager, temp_dirs):
        """Test creating a compressed backup."""
        chroma_dir, backup_dir = temp_dirs

        backup_path = manager.create_backup(description="Test backup")

        assert backup_path.exists()
        assert backup_path.suffix == ".gz"
        assert tarfile.is_tarfile(str(backup_path))

        # Verify backup contains ChromaDB data
        with tarfile.open(backup_path, "r:gz") as tar:
            members = tar.getmembers()
            member_names = [m.name for m in members]

            assert any("chroma_db" in name for name in member_names)
            assert "backup_metadata.json" in member_names

    def test_create_backup_uncompressed(self, temp_dirs):
        """Test creating an uncompressed backup."""
        chroma_dir, backup_dir = temp_dirs
        manager = BackupManager(
            chroma_dir=chroma_dir,
            backup_dir=backup_dir,
            max_backups=5,
            compress=False,
        )

        backup_path = manager.create_backup(description="Test backup")

        assert backup_path.exists()
        assert backup_path.is_dir()
        assert (backup_path / "chroma_db").exists()
        assert (backup_path / "backup_metadata.json").exists()

    def test_create_backup_custom_name(self, manager):
        """Test creating a backup with custom name."""
        backup_path = manager.create_backup(
            description="Custom backup",
            backup_name="my_custom_backup",
        )

        assert "my_custom_backup" in backup_path.name

    def test_create_backup_nonexistent_chroma_dir(self, temp_dirs):
        """Test backup creation with nonexistent ChromaDB directory."""
        _, backup_dir = temp_dirs
        manager = BackupManager(
            chroma_dir="/nonexistent/path",
            backup_dir=backup_dir,
        )

        with pytest.raises(FileNotFoundError):
            manager.create_backup()

    def test_list_backups_empty(self, manager):
        """Test listing backups when none exist."""
        backups = manager.list_backups()
        assert backups == []

    def test_list_backups(self, manager):
        """Test listing backups."""
        # Create multiple backups
        manager.create_backup(description="Backup 1", backup_name="backup_001")
        manager.create_backup(description="Backup 2", backup_name="backup_002")

        backups = manager.list_backups()

        assert len(backups) == 2
        assert all("backup_name" in b for b in backups)
        assert all("description" in b for b in backups)
        assert all("created_at" in b for b in backups)
        assert all("size_mb" in b for b in backups)

        # Should be sorted by creation date (newest first)
        assert backups[0]["backup_name"] == "backup_002"
        assert backups[1]["backup_name"] == "backup_001"

    def test_get_backup_metadata(self, manager):
        """Test extracting backup metadata."""
        backup_path = manager.create_backup(description="Test metadata")

        metadata = manager._get_backup_metadata(backup_path)

        assert metadata is not None
        assert metadata["description"] == "Test metadata"
        assert "created_at" in metadata
        assert "version" in metadata

    def test_get_backup_size(self, manager):
        """Test getting backup size."""
        backup_path = manager.create_backup()

        size_mb = manager._get_backup_size(backup_path)

        assert size_mb > 0
        assert isinstance(size_mb, float)

    def test_validate_backup_valid(self, manager):
        """Test validating a valid backup."""
        backup_path = manager.create_backup()

        assert manager._validate_backup(backup_path) is True

    def test_validate_backup_invalid(self, manager, temp_dirs):
        """Test validating an invalid backup."""
        _, backup_dir = temp_dirs

        # Create invalid backup (empty tar.gz)
        invalid_backup = backup_dir / "invalid.tar.gz"
        with tarfile.open(invalid_backup, "w:gz") as tar:
            pass  # Empty archive

        assert manager._validate_backup(invalid_backup) is False

    def test_restore_backup_compressed(self, temp_dirs):
        """Test restoring from compressed backup."""
        chroma_dir, backup_dir = temp_dirs

        # Create backup
        manager = BackupManager(chroma_dir=chroma_dir, backup_dir=backup_dir)
        backup_path = manager.create_backup()

        # Delete ChromaDB directory
        shutil.rmtree(chroma_dir)
        assert not chroma_dir.exists()

        # Restore backup
        manager.restore_backup(backup_path, force=True)

        # Verify restoration
        assert chroma_dir.exists()
        assert (chroma_dir / "chroma.sqlite3").exists()
        assert (chroma_dir / "data" / "vectors.bin").exists()

    def test_restore_backup_uncompressed(self, temp_dirs):
        """Test restoring from uncompressed backup."""
        chroma_dir, backup_dir = temp_dirs

        # Create uncompressed backup
        manager = BackupManager(
            chroma_dir=chroma_dir,
            backup_dir=backup_dir,
            compress=False,
        )
        backup_path = manager.create_backup()

        # Delete ChromaDB directory
        shutil.rmtree(chroma_dir)
        assert not chroma_dir.exists()

        # Restore backup
        manager.restore_backup(backup_path, force=True)

        # Verify restoration
        assert chroma_dir.exists()
        assert (chroma_dir / "chroma.sqlite3").exists()

    def test_restore_backup_nonexistent(self, manager):
        """Test restoring from nonexistent backup."""
        with pytest.raises(FileNotFoundError):
            manager.restore_backup("/nonexistent/backup.tar.gz")

    def test_restore_backup_invalid(self, manager, temp_dirs):
        """Test restoring from invalid backup."""
        _, backup_dir = temp_dirs

        # Create invalid backup
        invalid_backup = backup_dir / "invalid.tar.gz"
        with tarfile.open(invalid_backup, "w:gz") as tar:
            pass  # Empty archive

        with pytest.raises(ValueError):
            manager.restore_backup(invalid_backup, force=True)

    def test_restore_backup_without_force(self, manager, temp_dirs):
        """Test restore fails without force when data exists."""
        chroma_dir, _ = temp_dirs

        backup_path = manager.create_backup()

        # Try to restore without force (chroma_dir still exists)
        with pytest.raises(ValueError, match="Use force=True"):
            manager.restore_backup(backup_path, force=False)

    def test_delete_backup(self, manager):
        """Test deleting a backup."""
        backup_path = manager.create_backup()

        assert backup_path.exists()

        manager.delete_backup(backup_path)

        assert not backup_path.exists()

    def test_delete_backup_nonexistent(self, manager):
        """Test deleting nonexistent backup."""
        with pytest.raises(FileNotFoundError):
            manager.delete_backup("/nonexistent/backup.tar.gz")

    def test_retention_policy(self, temp_dirs):
        """Test backup retention policy."""
        chroma_dir, backup_dir = temp_dirs
        manager = BackupManager(
            chroma_dir=chroma_dir,
            backup_dir=backup_dir,
            max_backups=3,
        )

        # Create 5 backups (exceeds max_backups=3)
        for i in range(5):
            manager.create_backup(description=f"Backup {i}", backup_name=f"backup_{i:03d}")

        backups = manager.list_backups()

        # Should only have 3 backups (oldest deleted)
        assert len(backups) == 3

        # Should have the 3 newest backups
        backup_names = [b["backup_name"] for b in backups]
        assert "backup_004" in backup_names
        assert "backup_003" in backup_names
        assert "backup_002" in backup_names
        assert "backup_001" not in backup_names  # Deleted
        assert "backup_000" not in backup_names  # Deleted

    def test_retention_policy_unlimited(self, temp_dirs):
        """Test unlimited backup retention (max_backups=0)."""
        chroma_dir, backup_dir = temp_dirs
        manager = BackupManager(
            chroma_dir=chroma_dir,
            backup_dir=backup_dir,
            max_backups=0,  # Unlimited
        )

        # Create 5 backups
        for i in range(5):
            manager.create_backup(backup_name=f"backup_{i}")

        backups = manager.list_backups()

        # Should have all 5 backups
        assert len(backups) == 5

    def test_export_to_json(self, manager, temp_dirs):
        """Test exporting metadata to JSON."""
        _, backup_dir = temp_dirs
        output_file = backup_dir / "export.json"

        manager.export_to_json(output_file)

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "export_date" in data
        assert "chroma_dir" in data
        assert "version" in data

    def test_get_backup_stats(self, manager):
        """Test getting backup statistics."""
        # Create some backups
        manager.create_backup(backup_name="backup_1")
        manager.create_backup(backup_name="backup_2")

        stats = manager.get_backup_stats()

        assert stats["total_backups"] == 2
        assert stats["total_size_mb"] >= 0  # Can be 0 for very small test backups
        assert stats["max_backups"] == 5
        assert "backup_dir" in stats
        assert "oldest_backup" in stats
        assert "newest_backup" in stats

    def test_get_backup_stats_empty(self, manager):
        """Test backup stats when no backups exist."""
        stats = manager.get_backup_stats()

        assert stats["total_backups"] == 0
        assert stats["total_size_mb"] == 0
        assert stats["oldest_backup"] is None
        assert stats["newest_backup"] is None

    def test_backup_restore_roundtrip(self, temp_dirs):
        """Test complete backup and restore cycle."""
        chroma_dir, backup_dir = temp_dirs

        # Add more files to ChromaDB
        (chroma_dir / "test_file.txt").write_text("test content")
        subdir = chroma_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content")

        # Create backup
        manager = BackupManager(chroma_dir=chroma_dir, backup_dir=backup_dir)
        backup_path = manager.create_backup(description="Full roundtrip test")

        # Modify original data
        (chroma_dir / "test_file.txt").write_text("modified content")

        # Restore backup
        manager.restore_backup(backup_path, force=True)

        # Verify original content restored
        assert (chroma_dir / "test_file.txt").read_text() == "test content"
        assert (chroma_dir / "subdir" / "nested.txt").read_text() == "nested content"
