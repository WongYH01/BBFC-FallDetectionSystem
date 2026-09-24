package com.bbfc.notification.out.telegram.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record SendAnimationResponse(boolean ok, Result result) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Result(
            @JsonProperty("message_id") long messageId,
            Animation animation,
            Animation document) {

        /** Telegram returns a document instead of an animation when it declines to loop the file. */
        public String fileId() {
            if (animation != null) {
                return animation.fileId();
            }
            return document == null ? null : document.fileId();
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Animation(@JsonProperty("file_id") String fileId) {
    }
}
