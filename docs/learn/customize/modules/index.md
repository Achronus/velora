# Agent Modules - Crash Course

???+ api "API Docs"

    [`velora.models.nf.modules`](../../reference/models/modules.md)

There might be a time where you want to experiment with specific `Actor` or `Critic` modules individually, without using an existing agent.

While it's very uncommon, it's possible with our framework! You'll learn all about the different modules in this section. 😁

??? tip "In a Hurry?"

    Looking to jump to something specific? Use the navigation menu on the left 👈!

## The Basics

Under-the-hood, Velora's agents are made up of different modules that utilise PyTorch's functionality.

Each module has it's own `continuous` and `discrete` variant that is accessible from the API using:

```python
velora.models.nf.modules import [module]
```

???+ tip "Read the API Docs!"

    We highly recommend you refer to the API docs for complete details on each of the modules. They are far more extensive than the details in this chapter. API doc links are provided in the respective sections.

    This chapter is only a quick crash course for the methods the modules use and assumes that you already have a solid grounding in PyTorch basics.

## Saving and Loading Modules

Every module has it's own saving and loading mechanism. To save a modules state, we use the `state_dict` method:

```python
sd = module.state_dict()
```

And to restore it into a new or existing module, we use `load_state_dict`:

```python
module.load_state_dict(sd)
```

---

When you're ready, click the button 👇 to jump into our first set of modules - `ActorModules`! 🚀
