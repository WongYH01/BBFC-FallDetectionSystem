# Configuration

Two places, split by whether something is a secret.

| | File | Committed? | Holds |
|---|---|---|---|
| Secrets | `../.env.local`, `../.env.cloud` | no — copy from `.env.example.local` | Bot token, chat id, DB password, Supabase keys, host ports |
| Tunables | `config/application.yml` | **yes** | Message wording, clip wait and escalation timing, the polling and scheduler switches |

Defaults and the wiring between them live in `src/main/resources/application.properties`, which
is packaged into the jar. It maps `${TELEGRAM_BOT_TOKEN}` and friends onto the `telegram.*`
properties, so the same key names work whether they come from `.env` or from real environment
variables.

Precedence, lowest to highest: packaged `application.properties` → `config/application.yml` →
`.env` → real environment variables.

## Two modes

The database is either the local `db` container or a Cloud SQL instance. Same image, same jar —
only configuration differs.

```bash
docker compose --env-file .env.local --profile local up   # local Postgres + Storage
docker compose --env-file .env.cloud --profile cloud up   # Cloud SQL, no local containers
```

Every service is profiled, so a bare `docker compose up` starts nothing rather than reaching for
Cloud SQL by accident.

Cloud mode connects through the Cloud SQL JDBC socket factory: no proxy process, just a
`socketFactory=` parameter in `SPRING_DATASOURCE_URL`. It authenticates with your ADC file,
bind-mounted at `/creds.json` via `GCLOUD_ADC_PATH`. Refresh it with
`gcloud auth application-default login` when connections start failing with an auth error.

Clips follow the same split. `.env.cloud` sets `CLIPS_STORE=gcs` and `CLIPS_GCS_BUCKET`, and
compose defaults `CLIPS_STORE` to `supabase`, so each mode picks its store without a manual switch.
The GCS client uses the same ADC file as the database. The cloud bucket is created by hand, so
`.env.cloud` sets `CLIPS_ENSURE_BUCKET=false`; your account needs `roles/storage.objectAdmin` on
it. Both stores save the key as `{room_id}/{event_id}.mp4`, so a key means the same thing in
either mode.

## Running it

```bash
cp .env.example.local .env.local   # then fill in the blanks
docker compose --env-file .env.local --profile local up -d db
cp .env.local .env           # application.properties imports ./.env by name
./mvnw spring-boot:run
```

That copy is the one manual step: `spring.config.import` reads `./.env` literally, so a host run
never sees `.env.local` or `.env.cloud` directly. Treat `.env` as scratch — edit the mode file,
then copy.

Host runs resolve `localhost:${DB_PORT}` rather than `db:5432`, which is why `.env.local` leaves
`SPRING_DATASOURCE_URL` unset: each context falls through to its own correct default.

## Changing the alert wording

Edit `message-template` in `config/application.yml` and restart. No rebuild. The placeholders are
`{roomName}`, `{time}` and `{confidence}`; anything else fails at startup rather than rendering a
broken message. `triage-question-template` takes `{roomName}` and `{handledLine}`, and
`escalation-template` takes `{ordinal}` and `{roomName}`, under the same check.

## In a container

`.env` is absent from the image — the import is `optional:`, so Spring skips it and the same keys
arrive as injected environment variables instead. `config/application.yml` gets bind-mounted, so
tunables stay editable without rebuilding.

## Tests

Tests override all of the above via inlined properties on `AbstractIntegrationTest`, which outrank
both `.env` and real environment variables. That is deliberate: without it, a developer who has
exported `TELEGRAM_BOT_TOKEN` to run the app locally would have the test suite talking to the real
bot. `NotificationApplicationTests.doesNotLoadRealCredentials` fails loudly if that protection
ever breaks.
