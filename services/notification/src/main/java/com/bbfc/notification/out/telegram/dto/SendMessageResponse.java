package com.bbfc.notification.out.telegram.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record SendMessageResponse(boolean ok, Result result) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Result(@JsonProperty("message_id") long messageId) {
    }
}
