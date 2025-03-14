from sqlalchemy import Engine, ScalarResult
from sqlmodel import Session, SQLModel, create_engine, select

from velora.metrics.models import Episode


def get_db_engine() -> Engine:
    """
    Starts the metric database engine and returns it.

    Returns:
        engine (sqlalchemy.Engine): a database engine instance
    """
    from velora.metrics.models import Episode, Experiment, Step  # pragma: no cover

    engine = create_engine("sqlite:///metrics.db")
    SQLModel.metadata.create_all(engine)
    return engine


def get_current_episode(
    session: Session,
    experiment_id: int,
    current_ep: int,
) -> ScalarResult[Episode]:
    """
    Queries a database session to retrieve the current episode for an experiment.

    Parameters:
        session (sqlmodel.Session): a metric database session
        experiment_id (int): the current experiment's unique ID
        current_ep (int): the current episode index

    Returns:
        results (ScalarResult[Episode]): a iterable set of matching episodes.
    """
    statement = select(Episode).where(
        Episode.experiment_id == experiment_id,
        Episode.episode_num == current_ep,
    )
    return session.exec(statement)
