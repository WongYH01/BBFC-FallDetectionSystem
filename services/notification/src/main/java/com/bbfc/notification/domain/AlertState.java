package com.bbfc.notification.domain;

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
            case EXHAUSTED, TRIAGED -> false;
        };
    }
}


