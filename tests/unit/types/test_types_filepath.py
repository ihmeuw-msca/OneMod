import pytest
from onemod.types.filepath import FilePath

@pytest.mark.unit
def test_valid_filepath():
    model = FilePath(path="/path/to/file.csv", extension=".csv")
    assert model.path == "/path/to/file.csv"

@pytest.mark.unit
def test_invalid_filepath_extension():
    with pytest.raises(ValueError):
        FilePath(path="/path/to/file.txt", extension=".csv")  # Wrong extension
