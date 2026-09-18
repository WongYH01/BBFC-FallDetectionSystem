package com.bbfc.notification.core.service;

public class ClipTooLargeException extends RuntimeException {

    public ClipTooLargeException(long sizeBytes, long maxBytes) {
        super("Clip is " + sizeBytes + " bytes, over the " + maxBytes + " byte limit");
    }
}
