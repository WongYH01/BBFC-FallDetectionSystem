package com.bbfc.notification.core.domain;

public class IllegalAlertTransitionException extends RuntimeException {
    public IllegalAlertTransitionException(AlertState alertFrom, AlertState alertTo){
        super("Cannot transition alert from " + alertFrom + " to "+ alertTo);
    }
}
