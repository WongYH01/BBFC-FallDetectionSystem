package com.bbfc.notification.out.telegram;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.CallbackCommand;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.port.NotificationChannel;
import com.bbfc.notification.out.telegram.dto.SendMessageResponse;
import com.bbfc.notification.out.telegram.dto.TelegramRequests.AnswerCallbackQuery;
import com.bbfc.notification.out.telegram.dto.TelegramRequests.EditMessageText;
import com.bbfc.notification.out.telegram.dto.TelegramRequests.InlineKeyboardButton;
import com.bbfc.notification.out.telegram.dto.TelegramRequests.ReplyMarkup;
import com.bbfc.notification.out.telegram.dto.TelegramRequests.SendMessage;

@Component
public class TelegramBotClient implements NotificationChannel {

    private static final Logger log = LoggerFactory.getLogger(TelegramBotClient.class);
    private static final String PARSE_MODE = "HTML";

    private final RestClient restClient;
    private final String chatGroupId;

    public TelegramBotClient(TelegramConfig telegramConfig,  RestClient.Builder restClientBuilder){
        this.chatGroupId = telegramConfig.chatGroupId();
        this.restClient = restClientBuilder
                .baseUrl(telegramConfig.baseUrl() + "/bot" + telegramConfig.botToken())
                .build();
    }

    @Override
    public long sendAlert(EventId eventId, String message) {
        return send(message, ReplyMarkup.singleRow(new InlineKeyboardButton(
                "✅ Acknowledge",
                new CallbackCommand.Acknowledge(eventId).encode())));
    }

    @Override
    public long sendTriageQuestion(EventId eventId, String message) {
        return send(message, ReplyMarkup.singleRow(
                new InlineKeyboardButton("👍 Genuine",
                        new CallbackCommand.Triage(eventId, Outcome.GENUINE).encode()),
                new InlineKeyboardButton("🚫 False alarm",
                        new CallbackCommand.Triage(eventId, Outcome.FALSE_ALARM).encode())));
    }

    @Override
    public void closeMessage(long messageId, String text) {
        try {
            restClient.post()
                    .uri("/editMessageText")
                    .body(new EditMessageText(chatGroupId, messageId, text, PARSE_MODE, ReplyMarkup.none()))
                    .retrieve()
                    .toBodilessEntity();
        } catch (RestClientResponseException e) {
            // A redelivered callback re-edits a message to text it already has. Telegram rejects
            // that, and messages over 48h old can't be edited at all — neither is worth failing
            // the acknowledgement that triggered this.
            if (isAlreadyUpToDate(e)) {
                log.debug("Message {} already closed out", messageId);
            } else {
                log.warn("Could not close message {}: {}", messageId, e.getMessage());
            }
        } catch (RuntimeException e) {
            log.warn("Could not close message {}", messageId, e);
        }
    }

    @Override
    public void answerCallback(String callbackQueryId, String text) {
        try {
            restClient.post()
                    .uri("/answerCallbackQuery")
                    .body(new AnswerCallbackQuery(callbackQueryId, text))
                    .retrieve()
                    .toBodilessEntity();
        } catch (RuntimeException e) {
            log.warn("Could not answer callback {}", callbackQueryId, e);
        }
    }

    private long send(String message, ReplyMarkup replyMarkup) {
        SendMessageResponse response = restClient.post()
                .uri("/sendMessage")
                .body(new SendMessage(chatGroupId, message, PARSE_MODE, replyMarkup))
                .retrieve()
                .body(SendMessageResponse.class);

        if (response == null || !response.ok() || response.result() == null) {
            throw new IllegalStateException("Telegram rejected sendMessage: " + response);
        }
        return response.result().messageId();
    }

    private static boolean isAlreadyUpToDate(RestClientResponseException e) {
        return e.getResponseBodyAsString().contains("message is not modified");
    }
}
