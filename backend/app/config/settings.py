from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator

import os
from dotenv import dotenv_values


class Settings(BaseSettings):
    app_name: str = "FastAPI"

    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str

    # Database URI can be provided directly or built from the other variables.
    DATABASE_URI: Optional[str] = None

    @validator("DATABASE_URI", pre=True)
    def assemble_db_connection(cls, uri: Optional[str],
                               values: Dict[str, Any]) -> Any:
        """Builds the MariaDB connector if it is not provided."""

        # Checks if provided directly.
        if isinstance(uri, str):
            return uri

        return 'mariadb+mariadbconnector://%s:%s@%s:%s/%s?charset=utf8' % (
            values.get("DB_USER"),
            values.get("DB_PASSWORD"),
            values.get("DB_HOST"),
            values.get("DB_PORT"),
            values.get('DB_NAME'),
        )

    @classmethod
    def from_env(cls, **kwargs):
        """
        Tries to load from .env first if it exists, otherwise from
        the environment variables. Allows running the app locally
        (outside docker) without having to write env variables.
        """

        # Load sensitive variables from .env files and
        # override with environment variables
        env_vars = {
            **dotenv_values("../.env"),
            **dotenv_values("./fastapi.env"),
            **os.environ,
        }

        # Ensure that required variables are set or provide default values
        database_uri = cls.assemble_db_connection(None, env_vars)

        return cls(
            DATABASE_URI=database_uri,
            DB_NAME=env_vars.get("DATABASE_NAME", ""),
            DB_USER=env_vars.get("DATABASE_USER", ""),
            DB_PASSWORD=env_vars.get("DATABASE_PASSWORD", ""),
            DB_HOST=env_vars.get("DATABASE_HOST", ""),
            DB_PORT=env_vars.get("DATABASE_PORT", ""),
            **kwargs,
        )


class Config:
    case_sensitive = True


settings = Settings.from_env()
print(settings)
