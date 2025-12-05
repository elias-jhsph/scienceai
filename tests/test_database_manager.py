"""Tests for DatabaseManager functionality."""

from pathlib import Path


class TestDatabaseManagerBasics:
    """Test basic DatabaseManager functionality."""

    def test_sha256sum_function(self, temp_dir: Path) -> None:
        """Test SHA256 hash computation for files."""
        from scienceai.database_manager import sha256sum

        # Create a test file with known content
        test_file = temp_dir / "test_file.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Compute hash
        result = sha256sum(str(test_file))

        # Verify it's a valid hex string of correct length (64 chars for SHA256)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

        # Same content should produce same hash
        test_file2 = temp_dir / "test_file2.txt"
        test_file2.write_bytes(test_content)
        assert sha256sum(str(test_file2)) == result

    def test_get_projects_empty_directory(self, temp_dir: Path) -> None:
        """Test get_projects returns empty list for empty directory."""
        from scienceai.database_manager import get_projects

        # Create scienceai_db directory
        db_dir = temp_dir / "scienceai_db"
        db_dir.mkdir()

        projects = get_projects(str(temp_dir))
        assert projects == []

    def test_get_projects_with_projects(self, temp_dir: Path) -> None:
        """Test get_projects returns project names."""
        from scienceai.database_manager import get_projects

        # Create scienceai_db directory with project subdirectories
        db_dir = temp_dir / "scienceai_db"
        db_dir.mkdir()
        (db_dir / "project1").mkdir()
        (db_dir / "project2").mkdir()

        projects = get_projects(str(temp_dir))
        assert set(projects) == {"project1", "project2"}

    def test_get_projects_excludes_checkpoints(self, temp_dir: Path) -> None:
        """Test get_projects excludes checkpoint directories."""
        from scienceai.database_manager import get_projects

        # Create scienceai_db directory
        db_dir = temp_dir / "scienceai_db"
        db_dir.mkdir()
        (db_dir / "project1").mkdir()
        (db_dir / "project1_-checkpoint-_backup").mkdir()

        projects = get_projects(str(temp_dir))
        assert projects == ["project1"]
