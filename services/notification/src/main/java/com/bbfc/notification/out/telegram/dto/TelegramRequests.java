package com.bbfc.notification.out.telegram.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/** Wire shapes for the Telegram Bot API methods this service calls. */
public final class TelegramRequests {

    private TelegramRequests() {
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record SendMessage(
            @JsonProperty("chat_id") String chatId,
            String text,
            @JsonProperty("parse_mode") String parseMode,
            @JsonProperty("reply_markup") ReplyMarkup replyMarkup,
            @JsonProperty("reply_to_message_id") Long replyToMessageId,
            @JsonProperty("allow_sending_without_reply") Boolean allowSendingWithoutReply
    ) {
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record EditMessageText(
            @JsonProperty("chat_id") String chatId,
            @JsonProperty("message_id") long messageId,
            String text,
            @JsonProperty("parse_mode") String parseMode,
            @JsonProperty("reply_markup") ReplyMarkup replyMarkup
    ) {
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record EditMessageCaption(
            @JsonProperty("chat_id") String chatId,
            @JsonProperty("message_id") long messageId,
            String caption,
            @JsonProperty("parse_mode") String parseMode,
            @JsonProperty("reply_markup") ReplyMarkup replyMarkup
    ) {
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record AnswerCallbackQuery(
            @JsonProperty("callback_query_id") String callbackQueryId,
            String text
    ) {
    }

    public record ReplyMarkup(@JsonProperty("inline_keyboard") List<List<InlineKeyboardButton>> inlineKeyboard) {

        /** Telegram clears a message's keyboard when it is edited with an empty one. */
        public static ReplyMarkup none() {
            return new ReplyMarkup(List.of());
        }

        public static ReplyMarkup singleRow(InlineKeyboardButton... buttons) {
            return new ReplyMarkup(List.of(List.of(buttons)));
        }
    }

    public record InlineKeyboardButton(String text, @JsonProperty("callback_data") String callbackData) {
    }
}
