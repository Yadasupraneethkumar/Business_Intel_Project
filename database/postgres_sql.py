import os
from urllib.parse import quote

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()



def get_engine(database: str | None = None):
    """Create a PostgreSQL connection engine."""

    database = database or os.getenv("POSTGRES_DATABASE")

    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    required = {
        "POSTGRES_DATABASE": database,
        "POSTGRES_HOST": host,
        "POSTGRES_PORT": port,
        "POSTGRES_USER": user,
        "POSTGRES_PASSWORD": password,
    }

    missing = [
        name for name, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing PostgreSQL environment variables: "
            + ", ".join(missing)
        )

    encoded_user = quote(user, safe="")
    encoded_password = quote(password, safe="")

    connection_string = (
        f"postgresql+psycopg2://"
        f"{encoded_user}:{encoded_password}@{host}:{port}/{database}"
    )

    return create_engine(connection_string)


def create_database() -> None:
    """
    Create the target database if it does not exist.
    
    Uses psycopg2 directly with autocommit to bypass SQLAlchemy transaction wrapping.
    """
    target_db = os.getenv("POSTGRES_DATABASE")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    
    try:
        # Connect to default 'postgres' database using psycopg2 directly
        conn = psycopg2.connect(
            host=host,
            port=port,
            database="postgres",
            user=user,
            password=password
        )
        # Enable autocommit mode before executing CREATE DATABASE
        conn.autocommit = True
        
        cursor = conn.cursor()
        try:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (target_db,)
            )
            
            if not cursor.fetchone():
                print(f"Database '{target_db}' does not exist. Creating...")
                cursor.execute(f"CREATE DATABASE {target_db}")
                print(f"Database '{target_db}' created successfully.")
            else:
                print(f"Database '{target_db}' already exists.")
        finally:
            cursor.close()
            conn.close()
    except Exception as exc:
        print(f"Failed to create database: {exc}")
        raise