import pytest

from onemod.dtypes.filepath import FilePath


@pytest.mark.unit
def test_valid_filepath(tmp_path):
    test_file = tmp_path / "test_file.csv"
    test_file.touch()

    test_filepath = FilePath(path=test_file, extension=".csv")
    assert test_filepath.path == test_file


@pytest.mark.unit
def test_invalid_filepath_does_not_exist():
    with pytest.raises(ValueError):
        FilePath(path="/path/to/missing_file.csv", extension=".csv")


@pytest.mark.unit
def test_invalid_filepath_extension(tmp_path):
    test_file = tmp_path / "test_file.txt"
    test_file.touch()

    with pytest.raises(ValueError):
        FilePath(path=test_file, extension=".csv")  # Wrong extension
