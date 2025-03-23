import os
import pytest
from sqlmodel import Session, SQLModel, create_engine
from typing import Generator, Tuple


@pytest.fixture(name="engine")
def engine_fixture():
    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)

    # Create the tables
    SQLModel.metadata.create_all(engine)

    # Return the engine
    yield engine

    # Clean up
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="session")
def session_fixture(engine) -> Generator[Session, None, None]:
    with Session(engine) as session:
        # Begin a nested transaction
        session.begin_nested()

        # Return the session
        yield session

        # Rollback the transaction
        session.rollback()


@pytest.fixture(name="experiment")
def create_experiment_fixture(session: Session) -> Tuple[Session, int]:
    os.environ["VELORA_TEST_MODE"] = "True"

    from velora.metrics.models import Experiment

    # Create a test experiment
    experiment = Experiment(agent="TestAgent", env="TestEnv", config='{"test": true}')

    # Add and commit
    session.add(experiment)
    session.commit()

    # Refresh to get the ID
    session.refresh(experiment)

    # Return the session and experiment ID
    return session, experiment.id
