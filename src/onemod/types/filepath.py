from pydantic import BaseModel, FilePath as PydanticFilePath, model_validator


class FilePath(BaseModel):
    path: PydanticFilePath
    extension: str | None = None
    
    @model_validator(mode="after")
    def check_extension(self):
        path = self.path
        extension = self.extension
        if extension is not None and path.suffix != extension:
            raise ValueError(f"File extension '{path.suffix}' does not match expected extension '{extension}'")
        return self