CREATE TABLE alerts (
    event_id        VARCHAR(64)  PRIMARY KEY,
    room_id         VARCHAR(64)  NOT NULL,
    room_name       VARCHAR(128) NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    state           VARCHAR(24)  NOT NULL,
    acknowledged_by VARCHAR(64),
    acknowledged_at TIMESTAMPTZ,
    escalated_at    TIMESTAMPTZ,
    outcome         VARCHAR(16),
    outcome_at      TIMESTAMPTZ
);