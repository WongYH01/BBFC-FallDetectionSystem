package com.bbfc.notification.in.telegram.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/** The slice of Telegram's getUpdates payload this service reads. */
public final class TelegramUpdates {

    private TelegramUpdates() {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record GetUpdatesResponse(boolean ok, List<Update> result) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Update(
            @JsonProperty("update_id") long updateId,
            @JsonProperty("callback_query") CallbackQuery callbackQuery
    ) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CallbackQuery(
            String id,
            TelegramUser from,
            String data,
            // Since Bot API 7.0 this may be an inaccessible message, so treat it as optional.
            Message message
    ) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record TelegramUser(
            long id,
            String username,
            @JsonProperty("first_name") String firstName,
            @JsonProperty("last_name") String lastName
    ) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Message(Chat chat) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Chat(long id) {
    }
}
