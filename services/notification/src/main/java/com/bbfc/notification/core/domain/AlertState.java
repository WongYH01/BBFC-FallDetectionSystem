package com.bbfc.notification.core.domain;

public enum AlertState {
    DISPATCHED,
    ESCALATING,
    ACKNOWLEDGED,
    EXHAUSTED,
    TRIAGED;

    public boolean canTransitionTo(AlertState alertState){
        return switch (this){
            case DISPATCHED -> alertState == ESCALATING || alertState == ACKNOWLEDGED;
            case ESCALATING -> alertState == ESCALATING || alertState == ACKNOWLEDGED || alertState == EXHAUSTED;
            case ACKNOWLEDGED -> alertState == TRIAGED;
            // Escalation gave up, but the buttons stay live in the chat and a late tap is the
            // common ward case — recording who finally handled it beats a dead button.
            case EXHAUSTED -> alertState == ACKNOWLEDGED;
            case TRIAGED -> false;
        };
    }
}


