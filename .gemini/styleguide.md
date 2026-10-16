# BBFC Fall Detection System: Review Guide

A fall-detection system for elderly care. A missed or lost alert is the worst
possible failure. Review for correctness and safety first, style last.

## Severity
- CRITICAL: anything that can drop, duplicate or delay a fall alert; secrets
  (tokens, keys, .env values) in code, logs, image layers or workflow output.
- HIGH: unhandled errors on alert/clip paths, race conditions, missing input
  validation at REST/Telegram boundaries, non-root/healthcheck removed from a
  Dockerfile, unpinned versions (actions, pip, apt, base images).
- MEDIUM: missing tests for new logic, dead code, needless complexity.
- LOW: naming and formatting. Do not comment on these unless asked.

## General
- Prefer the smallest change that fully solves the problem. Flag new
  abstractions, options or dependencies that nothing needs yet.
- Code comments only when they explain a non-obvious *why*. Flag comments that
  restate what the code does.
- Pin every version: GitHub Actions tags, pip requirements, Docker base images.

## services/notification (Java, Spring Boot)
- Hexagonal layout: `core/domain` and `core/service` must not import from
  `in/`, `config/` or any framework-specific adapter; adapters talk to the core
  through `core/port` interfaces.
- Alert state changes go through `Alert`'s transitions; flag code that sets
  state directly or bypasses `IllegalAlertTransitionException`.
- Use the injected `Clock`, never `Instant.now()`/`LocalDateTime.now()`.
- Map domain exceptions to HTTP responses in `ApiExceptionHandler`, not in
  controllers.

## services/detection (Python 3.12)
- Never block or crash the detection loop on a notification failure; alerts go
  through the outbox and must survive a restart.
- The image is CPU-only and uses `opencv-python-headless`; flag anything that
  pulls CUDA wheels or the GUI OpenCV build.
- Use `logging`, not `print`.

## CI / Docker
- Secrets reach builds only as BuildKit secrets or GitHub secrets, never as
  build args or ENV.
- Workflows: least-privilege `permissions:` per job.