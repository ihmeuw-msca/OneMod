from pydantic import BaseModel, FilePath as PydanticFilePath, field_validator
from pathlib import Path

class FilePath(BaseModel):
    path: PydanticFilePath
    extension: str | None = None
    
    @field_validator('path')
    def check_extension(cls, value: Path, values: dict) -> Path:
        ext = values.get('extension')
        if ext is not None and value.suffix != ext:
            raise ValueError(f"File must have the {ext} extension.")
        return value
