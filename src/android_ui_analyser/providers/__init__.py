"""Pluggable perception providers (OCR / detection / grounding).

Concrete providers live in the ``ocr/``, ``detection/`` and ``grounding/`` subpackages
and self-register via the ``register_*`` decorators in :mod:`.registry`. They are
auto-discovered (``registry.load_providers``); no edits here are needed to add one.
"""
