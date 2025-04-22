# Working with Training Metrics

Understanding how your agent is learning is extremely important for figuring out how to optimize its performance and also fix it when it's broken.

To do this offline, we use a [SQLite [:material-arrow-right-bottom:]](https://www.sqlite.org/) database for storing our metrics called `metrics.db`.

## What's Inside It?

We've split the database into two main tables:

- [`experiment`](#experiments) - for tracking basic information about your experiment.
- [`episode`](#episodes) - for storing episode metrics.

### Experiments

???+ api "API Docs"

    [`velora.metrics.Experiment`](../reference/metrics.md#velora.metrics.Experiment)

The `experiment` table is the simplest and is primarily used to maintain an `id` for different experiments. It stores the following details:

- `id` - a unique identifier for each experiment.
- `agent` - the class name of the agent used.
- `env` - the name of the environment used during training.
- `config` - the agent's configuration details stored in a JSON format (the same one when saving a model!).
- `created_at` - the time and date when the experiment was created.

### Episodes

???+ api "API Docs"

    [`velora.metrics.Episode`](../reference/metrics.md#velora.metrics.Episode)

The `episode` table stores the metrics for each training episode. It's the most comprehensive table of the three and stores the following details:

- `id` - a unique identifier for the episode.
- `experiment_id` - the experiment ID associated to the episode.
- `episode_num` - the episode index.
- `reward` - the episodic reward (return).
- `length` - the number of timesteps performed to terminate the episode.
- `reward_moving_avg` - the episodes reward moving average based on the training `window_size`.
- `reward_moving_std` - the episodes reward moving standard deviation based on the training `window_size`.
- `actor_loss` - the average Actor loss for the episode.
- `critic_loss` - the average Critic loss for the episode.
- `entropy_loss` - the average Entropy loss for the episode.
- `created_at` - the date and time when the the entry was created.

## Exploring the Database

Once you've run a training instance with an agent's `train()` method, the database will automatically store the above metrics. You can then freely access them for your own analysis whenever you want! 😊

If you want to quickly explore the database, we recommend you use [DB Browser [:material-arrow-right-bottom:]](https://sqlitebrowser.org/). It's GUI interface is extremely useful for quickly checking the stored data and running SQL queries.

## Interacting With It

To get data out of the database and use it in your projects we can use a helper method to quickly get the `engine` and then build a session.

We use [SQLModel [:material-arrow-right-bottom:]](https://sqlmodel.tiangolo.com/) under the hood, so we can interact with our database tables [Pydantic [:material-arrow-right-bottom:]](https://docs.pydantic.dev/latest/) model style! 😍

We can use a `Session` as a context manager (recommended):

```python
from velora.metrics import get_db_engine, Episode
from sqlmodel import Session, select

engine = get_db_engine()

# Create a new session
with Session(engine) as session:
    # Get some data
    statement = select(Episode).where(
        Episode.experiment_id == 1
    )
    results = session.exec(statement)

    # Return specific elements
    for episode in results:
        print(episode.reward, episode.length)
```

This code should work 'as is'.

Or, as an instance:

```python
from velora.metrics import get_db_engine, Episode
from sqlmodel import Session, select

engine = get_db_engine()

# Create a new session
session = Session(engine)

# Get some data
statement = select(Episode).where(
    Episode.experiment_id == 1
)
results = session.exec(statement)

# Return specific elements
for episode in results:
    print(episode.reward, episode.length)

# Close the session
session.close()
```

This code should work 'as is'.

We highly recommend you read the [SQLModel [:material-arrow-right-bottom:]](https://sqlmodel.tiangolo.com/) documentation for more details.

---

Next, we'll dive into the utility methods for `Gymnasium` 👋.
