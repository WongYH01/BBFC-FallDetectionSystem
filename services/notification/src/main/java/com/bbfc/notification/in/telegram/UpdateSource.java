package com.bbfc.notification.in.telegram;

/**
 * Where callback taps come from. Long polling today; a webhook implementation would set the
 * webhook on {@code start()} and delete it on {@code stop()}, feeding the same handler.
 */
public interface UpdateSource {
    void start();
    void stop();
}
