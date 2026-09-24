package com.bbfc.notification.in.rest.dto;

public record ClipResponse(String eventId, String status, boolean sentToTelegram) {
}
