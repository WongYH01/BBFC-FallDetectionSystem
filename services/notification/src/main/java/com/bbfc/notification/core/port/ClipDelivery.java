package com.bbfc.notification.core.port;

/** What Telegram gives back after accepting a clip: its cached file id and the message id. */
public record ClipDelivery(String fileId, long messageId) {
}
