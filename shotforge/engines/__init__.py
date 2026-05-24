"""Pluggable engines — content-free tools, one per pipeline stage.

An engine takes a *spec* (prompt / refs / loras / paths) and produces output. It
embeds NO images, LoRAs, or prompts — those come from the asset libraries via the
composer. A project selects engines by name (project.yaml ``engines:``); the
registry lazily imports and instantiates them. Swap a model = swap an engine.
"""
