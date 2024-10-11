from pathlib import Path

from polars import DataFrame, read_parquet


class DataIOHandler:
    """
    Handler for reading and writing data in Parquet format.
    
    Notes
    _____
    * Future: Add support for other file formats.
    * TODO: expanding this vs. using pplkit.DataInterface
    """
    supported_formats: set[str] = {'parquet'}
    
    @staticmethod
    def read_data(path: Path) -> DataFrame:
        """Read data from a Parquet file and return it as a Polars DataFrame."""
        if path.suffix.lstrip('.') not in DataIOHandler.supported_formats:
            raise ValueError(f"Unsupported file format {path.suffix}.")
        
        try:
            df = read_parquet(path)
        except Exception as e:
            raise IOError(f"Failed to read data from {path}: {e}")
        
        return df

    @staticmethod
    def write_data(df: DataFrame, path: Path) -> None:
        """Write a Polars DataFrame to a Parquet file."""
        try:
            df.write_parquet(path)
        except Exception as e:
            raise IOError(f"Failed to write data to {path}: {e}")
